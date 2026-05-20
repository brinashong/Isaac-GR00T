# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0

# I2R implement
def _batch_error(
    preds: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor | None = None
) -> dict:  
    """Compute L1 error, L2 / MSE error, and Delta

    Args:
        preds: Predicted batched sequences of actions``.
        inputs: Given action outputs for the input sequences.

    Returns:
        Tuple of L1, L2 / MSE error and delta computed
    """

    if scale is None:
        scale = torch.ones(1, 1, preds.shape[-1], device=preds.device, dtype=preds.dtype)

    diff = (preds - gt) / scale
    l1_error = (diff.abs() * mask).sum() / (mask.sum() + 1e-6)
    mse_error = ((diff ** 2) * mask).sum() / (mask.sum() + 1e-6)
    
    if preds.shape[1] > 1:
        delta = (preds[:, 1:] - preds[:, :-1]) / scale
        gt_delta = (gt[:, 1:] - gt[:, :-1]) / scale
        mask_delta = mask[:, 1:] * mask[:, :-1]
        delta_mean = (delta.abs() * mask_delta).sum() / (mask_delta.sum() + 1e-6)
        if (mask_delta > 0).any():
            delta_max = (delta - gt_delta).abs()[mask_delta > 0].max()
        else:
            delta_max = torch.tensor(0.0, device=preds.device)
        delta_error = ((delta - gt_delta).abs() * mask_delta).sum() / (mask_delta.sum() + 1e-6)
    else:
        zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
        delta_mean = delta_max = delta_error = zero
    
    # return l1_error, l2_error, delta
    # if delta_mean is low but delta_error is high, model is under moving. If both deltas are low, then model is achieving ideal behaviour
    return {
        "l1_error": l1_error,
        "mse_error": mse_error,
        "delta_mean": delta_mean,   # mean predicted step size (how much the model moves)
        "delta_max": delta_max,     # worst-case delta error vs gt (where it diverges most)
        "delta_error": delta_error, # mean delta error vs gt (tracks gt dynamics quality)
    }

def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        custom_args: Any = None,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.custom_args = custom_args
        self.embodiment_tag = custom_args.embodiment_tag_list[0]
        # or use embodiment_id from inputs
        if self.custom_args.use_stats_norm_scale:
            # logging.info(f"embodiment_tag_list: {self.embodiment_tag}")
            # logging.info(f"loaded action_std: {str(self.custom_args.stats)}")
            # action_std = torch.tensor(self.custom_args.stats[self.embodiment_tag]['action']['std'], dtype=torch.float32)
            try:
                all_std = []
                for joint, stats in self.custom_args.stats[self.embodiment_tag]["action"].items():
                    all_std.extend(stats["std"])
                action_std = torch.tensor(all_std, dtype=torch.float32)
                min_std = 1e-3
                self.action_std = action_std.clamp_min(min_std).view(1, 1, -1)
                logging.info(f"{self.embodiment_tag} embodiment action_std loaded: {action_std}")
            except:
                self.custom_args.use_stats_norm_scale = False
                logging.info("WARNING: no action scale found from dataset. Setting use_stats_norm_scale to False to use batch-wise scale.")

        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        if (
           self.state.global_step % self.args.logging_steps == 0
           and model.training
           and "labels" in inputs
        ):
           if self.action_offset is not None:
               preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
           else:
               preds = outputs.logits.detach().argmax(dim=-1).cpu()
           with torch.no_grad():
               acc_local = _batch_accuracy(
                   preds, inputs["labels"].to(device=preds.device), self.action_offset
               )
           acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
           acc_mean = self._nested_gather(acc_tensor).mean().item()

           if self.args.local_rank in (-1, 0):
               self.log({"train_accuracy": acc_mean})

               # Log a sample of ground-truth vs predicted action tokens from
               # the first batch element so users can verify the model is
               # learning the right behaviors.
               shifted_labels = inputs["labels"][:1, 1:].cpu()
               shifted_preds = preds[:1, :-1]
               mask_0 = shifted_labels[0] != -100
               gt_tokens = shifted_labels[0][mask_0][:20]
               if self.action_offset is not None:
                   gt_tokens = gt_tokens - self.action_offset
               gt_sample = gt_tokens.tolist()
               pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
               logging.info(
                   "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                   "  GT:   %s\n  Pred: %s",
                   self.state.global_step,
                   gt_sample,
                   pred_sample,
               )

        # return (loss, outputs) if return_outputs else loss

        ### 20 May 2026 implement
        pred_actions = outputs['pred_actions']           # predicted velocity
        noisy_trajectory = outputs["noisy_trajectory"]   # [B, H, action_dim]
        t_cont = outputs["t_cont"]                       # [B]
        t_broadcast = t_cont[:, None, None]
        # Reconstruct predicted clean actions via one-step flow ODE
        #    x_1_pred = x_t + (1 - t) * v_pred
        pred_clean = noisy_trajectory + (1.0 - t_broadcast) * pred_actions # linear flow matching
        action_mask = outputs['action_mask']

        gt_actions = inputs['inputs']["action"].to(device=pred_actions.device)
        input_action_mask = inputs['inputs']['action_mask'].to(pred_actions.device).float()
        H = min(gt_actions.shape[1], pred_clean.shape[1])
        pred = pred_actions[:, :H, :]
        gt = gt_actions[:, :H, :]
        mask = input_action_mask[:, :H, :] # same as outputs['action_mask']
        valid = mask > 0
        if self.custom_args.use_stats_norm_scale:
            # Take pre-computed std from dataset stats.json
            scale = torch.ones((1, 1, pred.shape[-1]), device=pred.device, dtype=pred.dtype)
            scale[..., :self.action_std.shape[-1]] = self.action_std.to(pred.device)
            scale = scale.clamp(min=1e-2)
            scale = torch.where(scale < 2e-2, torch.ones_like(scale), scale)
        else: # compute batch-wise scale / std
            if valid.any():
                den = mask.sum(dim=(0,1)).clamp_min(1e-6)
                mean = (gt * mask).sum(dim=(0,1)) / den
                var = (((gt - mean) ** 2) * mask).sum(dim=(0,1)) / den
                scale = var.sqrt().clamp(min=1e-2).view(1,1,-1)
                scale = torch.where(scale < 2e-2, torch.ones_like(scale), scale)
            else:
                scale = torch.ones((1, 1, pred.shape[-1]), device=pred.device, dtype=pred.dtype)

        ### DEBUG ###
        # print("scale_pad: ", scale)
        # print("len scale_pad: ", len(scale[0][0]))
        # print("shape scale_pad: ", len(scale.shape))

        ### Smoothness (L2) - penalise jerk
        # L = mean over valid pairs of  || (a_{t+1} - a_t) / scale ||^2
        if pred.shape[1] > 1:
            delta = (pred[:, 1:] - pred[:, :-1]) / scale
            mask_delta = mask[:, 1:] * mask[:, :-1]
            smoothness_loss = (delta ** 2 * mask_delta).sum() / (mask_delta.sum() + 1e-6)
        else:
            smoothness_loss = torch.tensor(0.0, device=pred.device)

        ### Acceleration (L2) - penalise curvature
        if pred.shape[1] > 2:
            delta2 = (pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) / scale
            mask_delta2 = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
            accel_loss = (delta2 ** 2 * mask_delta2).sum() / (mask_delta2.sum() + 1e-6)
        else:
            accel_loss = torch.tensor(0.0, device=pred.device)

        ### Continuity
        # prev_action is last executed action, and not last predicted action
        mask0 = mask[:, 0, :]
        if self.custom_args.use_prev_action_conditioning:
            if False: # self.custom_args.use_multi_embodiment: 
            # TO-DO: cross-embodiment loss function compute, change index range to dynamic or as input (DO NOT HARDCODE)
                prev_action = inputs['inputs']['state'][:, 0, 58:87].to(pred.device)
                losses = []
                for b in range(pred.shape[0]):
                    valid_idx = mask[b, 0].bool().nonzero(as_tuple=True)[0]
                    pred_b = pred[b, 0, valid_idx]
                    # match first k dims of prev_action
                    prev_target_b = prev_action[b][:pred_b.shape[0]]
                    losses.append(torch.nn.functional.mse_loss(pred_b, prev_target_b, reduction='mean'))
                continuity_loss = torch.stack(losses).mean()

            else: # single embodiment
                # TO-DO: change index range to dynamic or as input (DO NOT HARDCODE)
                prev_action = inputs['inputs']['state'][:, 0, 58:87].to(pred.device)
                prev_action_pad = torch.zeros_like(pred[:, 0, :], dtype=pred.dtype)
                prev_action_pad[:, :prev_action.shape[-1]] = prev_action
                continuity_loss = (((pred[:, 0, :] - prev_action_pad) ** 2) * mask0).sum() / (mask0.sum() + 1e-6)
                ### NOTE: the following line computing valid_idx ASSUMES that all samples in batch share the SAME embodiment mask! 
                ## Which is meant for single embodiment OR dataloader able to group by embodiment 
                # valid_idx = mask[0, 0].bool().nonzero(as_tuple=True)[0] 
                # assert valid_idx.numel() == prev_action.shape[-1]
                # pred_first_valid = pred[:, 0, valid_idx]
                # continuity_loss = torch.nn.functional.mse_loss(pred_first_valid, prev_action)
        else:
            continuity_loss = (((pred[:, 0, :] - gt[: , 0, :]) ** 2) * mask0).sum() / (mask0.sum() + 1e-6)


        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0 # on rank 0
            and model.training
            and "action" in inputs['inputs']
        ):
            with torch.no_grad():
                metrics = _batch_error(pred, gt, mask, scale)

            log_dict = {
                "l1_error": metrics["l1_error"].item(),
                "mse_error": metrics["mse_error"].item(),
                "delta_mean": metrics["delta_mean"].item(),
                "delta_max": metrics["delta_max"].item(),
                "delta_error": metrics["delta_error"].item(),
                "action_loss": loss.item(), # loss computed from Transformer trainer class. Is this CrossEntropyLoss or L2 regression MSE loss? 
                "smoothness_loss": smoothness_loss.item(),
                "smoothness_loss_weighted": (self.custom_args.lambda_smooth * smoothness_loss).item(),
                "accel_loss": accel_loss.item(),
                "accel_loss_weighted": (self.custom_args.lambda_accel * accel_loss).item(),
                "continuity_loss": continuity_loss.item(),
                "continuity_loss_weighted": (self.custom_args.lambda_continuity * continuity_loss).item(),
                "t_cont_mean": t_cont.mean().item(), # noise levels that were sampled
            }

            if self.args.local_rank in (-1, 0):
                self.log(log_dict)

        total_loss = loss + self.custom_args.lambda_smooth * smoothness_loss + self.custom_args.lambda_accel * accel_loss + self.custom_args.lambda_continuity * continuity_loss
        self.loss = total_loss.detach().item()

        return (total_loss, outputs) if return_outputs else total_loss


        ### DEBUG ### 
        # print("outputs.keys: ", outputs.keys())
        ### outputs.keys:  dict_keys(['loss', 'action_loss', 'action_mask', 'backbone_features', 'state_features', 'pred_actions'])
        # print("inputs['inputs'].keys: ", inputs['inputs'].keys())
        ### inputs['inputs'].keys:  dict_keys(['state', 'action', 'embodiment_id', 'action_mask', 'input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])
        # print("loss: ", loss)
        # print("pred_actions: ", outputs['pred_actions'][0])
        # print("shape pred_actions: ", outputs['pred_actions'].shape)
        # print("len pred_actions[0][0]: ", outputs['pred_actions'][0].shape) # 12
        # print("input_actions: ", inputs['inputs']["action"][0])
        # print("shape input_actions: ", inputs['inputs']["action"].shape)
        # print("H: ", inputs['inputs']["action"].shape[1])
        # print("D: ", inputs['inputs']["action"].shape[-1])
        # print("input action mask: ", inputs['inputs']['action_mask'])
        # print("input action mask[0]: ", inputs['inputs']['action_mask'][0])
        # print("input action mask[1][0]: ", inputs['inputs']['action_mask'][1][0])
        # print("input action mask shape: ", inputs['inputs']['action_mask'].shape)
        # print("len input_actions[0][0]: ", inputs['inputs']["action"][0].shape)
        # print("prev_action: ", prev_action_pad.shape)
        # print("pred_masked[0:29]: ", (pred * mask)[:, 0, :][0][0:29])
        # print("pred_masked[29:58]: ", (pred * mask)[:, 0, :][0][29:58])
        # print("pred_masked[58:87]: ", (pred * mask)[:, 0, :][0][58:87])
        # print("pred_masked[87:128]: ", (pred * mask)[:, 0, :][0][87:128])


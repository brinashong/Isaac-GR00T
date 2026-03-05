from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

booster_config = {
  "video": ModalityConfig(
    delta_indices=[0],
    modality_keys=["color"], #"depth"],
  ),
  "state": ModalityConfig(
    delta_indices=[0],
    modality_keys=[
      "neck",
      "left_arm",
      "right_arm",
      "waist",
      "left_leg",
      "right_leg",
    ],
  ),
  "action": ModalityConfig(
    delta_indices=list(range(0, 16)),
    modality_keys=[
      "neck",
      "left_arm",
      "right_arm",
      "waist",
      "left_leg",
      "right_leg",
    ],
    action_configs=[
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
    ],
  ),
  "language": ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.action.task_description"],
  ),
}

register_modality_config(booster_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
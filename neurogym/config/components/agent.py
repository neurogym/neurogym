from typing import Literal

from neurogym.config.base import ConfigBase
from neurogym.config.components.trigger import TriggerConfig


class AgentTrainingConfig(TriggerConfig):
    """Training configuration for an agent.

    Defaults to 100 trials.
    """

    trigger: Literal["trials", "steps"] = "trials"
    value: int = 100


class AgentInferenceConfig(TriggerConfig):
    """Inference configuration for an agent.

    Defaults to 10000 steps.
    """

    trigger: Literal["trials", "steps"] = "steps"
    value: int = 10000


class AgentConfig(ConfigBase):
    """Top-level configuration for agent training and inference.

    Includes separate settings for training and inference.
    """

    training: AgentTrainingConfig = AgentTrainingConfig()
    inference: AgentInferenceConfig = AgentInferenceConfig()

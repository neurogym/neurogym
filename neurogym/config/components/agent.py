from typing import Literal

from pydantic import PositiveInt

from neurogym.config.base import ConfigBase


class AgentTrainingConfig(ConfigBase):
    """Training configuration for an agent.

    Attributes:
        unit: Unit for training duration.
        value: Number of units to train for, based on the unit.
    """

    unit: Literal["trial", "step"] = "trial"
    value: PositiveInt = 100


class AgentInferenceConfig(ConfigBase):
    """Inference configuration for an agent.

    Attributes:
        unit: Unit for inference duration.
        value: Number of units to infer for, based on the unit.
    """

    unit: Literal["trial", "step"] = "step"
    value: PositiveInt = 10000


class AgentConfig(ConfigBase):
    """Top-level configuration for agent training and inference.

    Includes separate settings for training and inference.
    """

    training: AgentTrainingConfig = AgentTrainingConfig()
    inference: AgentInferenceConfig = AgentInferenceConfig()

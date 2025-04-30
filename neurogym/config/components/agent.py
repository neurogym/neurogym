from neurogym.config.base import ConfigBase


class AgentTrainingConfig(ConfigBase):
    """Configuration options for agents during training.

    trials: Number of trials (episodes).
    steps: The number of training steps.
    """

    trials: int = 100
    steps: int = 100000


class AgentInferenceConfig(ConfigBase):
    """Configuration options for agents during inference.

    This is used when evaluating the agent's performance after training.

    trials: Number of trials (episodes).
    steps: The number of inference steps.
    """

    trials: int = 100
    steps: int = 10000


class AgentConfig(ConfigBase):
    """Configuration for agent-related options.

    training: Options related to training (see `AgentTrainingConfig`).
    inference: Options related to inference (see `AgentInferenceConfig`).
    """

    training: AgentTrainingConfig = AgentTrainingConfig()
    inference: AgentInferenceConfig = AgentInferenceConfig()

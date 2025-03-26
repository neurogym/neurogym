from neurogym.conf.base import ConfBase


class AgentTrainingConf(ConfBase):
    """Configuration options for agents during training.

    trials: Number of trials (episodes).
    steps: The number of training steps.
    """

    trials: int = 100
    steps: int = 100000


class AgentInferenceConf(ConfBase):
    """Configuration options for agents during inference.

    This is used when evaluating the agent's performance after training.

    trials: Number of trials (episodes).
    steps: The number of inference steps.
    """

    trials: int = 100
    steps: int = 10000


class AgentConf(ConfBase):
    """Configuration for agent-related options.

    training: Options related to training (cf. :ref:`AgentTrainingConf`).
    inference: Options related to inference (cf. :ref:`AgentInferenceConf`).
    """

    training: AgentTrainingConf = AgentTrainingConf()
    inference: AgentInferenceConf = AgentInferenceConf()

import enum

from neurogym.config.base import StrEnum


class EnvParam(StrEnum):
    """Types of environment parameters.

    Used for monitoring.
    """

    Reward = enum.auto()
    Action = enum.auto()
    Observation = enum.auto()


class NetParam(StrEnum):
    """Types of network parameters.

    Used for monitoring.
    """

    ActivationThreshold = enum.auto()
    MembranePotential = enum.auto()
    Activation = enum.auto()
    Weight = enum.auto()
    Bias = enum.auto()

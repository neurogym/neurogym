import enum
import sys

# HACK: Revise this if the minimal Python version
# is bumped to >= 3.11
if sys.version_info[:2] < (3, 11):

    class StrEnum(str, enum.Enum):
        pass

else:
    StrEnum = enum.StrEnum


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

import enum
import sys

# HACK: Revise this if the minimal Python version
# is bumped to >= 3.11
if sys.version_info[:2] <= (3, 10):

    class StrEnum(str, enum.Enum):
        pass

else:
    StrEnum = enum.StrEnum


# ==============================================================================
# Enums that can be used throughout NeuroGym instead of plain strings.
# ==============================================================================
class EnvParam(StrEnum):
    Reward = enum.auto()
    Action = enum.auto()
    Observation = enum.auto()


class NetParam(StrEnum):
    ActivationThreshold = enum.auto()
    MembranePotential = enum.auto()
    Activation = enum.auto()
    Weight = enum.auto()
    Bias = enum.auto()


class MonitorPhase(StrEnum):
    Training = enum.auto()
    Validation = enum.auto()
    Evaluation = enum.auto()


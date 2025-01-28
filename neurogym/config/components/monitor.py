# --------------------------------------
import enum

# --------------------------------------
from typing import Literal
from typing import Callable

# --------------------------------------
from pathlib import Path

# --------------------------------------
from neurogym import utils
from neurogym.config.base import ConfBase
from neurogym.config.components.log import LogConf
from neurogym.config.components.paths import LOCAL_DIR

class EnvParam(enum.StrEnum):
    Reward = enum.auto()
    Action = enum.auto()
    Observation = enum.auto()


class NetParam(enum.StrEnum):
    ActivationThreshold = enum.auto()
    MembranePotential = enum.auto()
    Activation = enum.auto()
    Weight = enum.auto()
    Bias = enum.auto()


class MonitorPlotConf(ConfBase):
    """
    Configuration options related to plotting.
    """

    save: bool = True
    ext: str = "png"
    interval: int = 100


class MonitorConf(ConfBase):
    """
    Monitor configuration.
    """

    title: str = "Monitor"
    steps: int = 100000
    trials: int = 1000
    dt: int = 10
    trigger: Literal["trial", "timestep"] = "trial"
    func: Callable = lambda: None
    save_dir: Path = LOCAL_DIR / f"monitor/behaviour/{utils.timestamp()}"
    plot: MonitorPlotConf = MonitorPlotConf()

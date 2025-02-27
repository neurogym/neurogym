from typing import Literal

from neurogym.config.base import ConfBase


class MonitorPlotConf(ConfBase):
    """Monitor configuration options related to plotting."""

    title: str = "NeuroGym"
    save: bool = True
    ext: str = "png"
    interval: int = 10


class MonitorConf(ConfBase):
    """Monitor configuration."""

    trigger: Literal["trial", "step"] = "trial"
    plot: MonitorPlotConf = MonitorPlotConf()

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import neurogym as ngym
from neurogym.config.base import ConfBase


class MonitorPlotConf(ConfBase):
    """Configuration options related to plotting."""

    save: bool = True
    ext: str = "png"
    interval: int = 100


class MonitorConf(ConfBase):
    """Monitor configuration."""

    title: str = "Monitor"
    steps: int = 100000
    trials: int = 1000
    dt: int = 10
    trigger: Literal["trial", "timestep"] = "trial"
    func: Callable = lambda: None
    save_dir: Path = ngym.LOCAL_DIR / f"monitor/behaviour/{ngym.utils.timestamp()}"
    plot: MonitorPlotConf = MonitorPlotConf()

from typing import Literal

from pydantic import PositiveInt

from neurogym.config.base import ConfigBase


class MonitorPlotConfig(ConfigBase):
    """Plotting configuration during monitoring.

    Attributes:
        create: Whether to generate plots.
        step: Number of steps to visualize on the figure.
        title: Title to display on plots.
        ext: File extension for saved plot images.
    """

    create: bool = True
    step: PositiveInt = 10
    title: str = ""
    ext: str = "png"


class MonitorConfig(ConfigBase):
    """Top-level configuration for monitoring.

    Attributes:
        name: Optional monitor name.
        trigger: Unit used to trigger saving output.
        interval: Number of trigger units between each save operation.
        verbose: Whether to enable verbose output.
        plot: Plotting configuration.
    """

    name: str = ""
    trigger: Literal["trial", "step"] = "trial"
    interval: PositiveInt = 1000
    verbose: bool = True
    plot: MonitorPlotConfig = MonitorPlotConfig()

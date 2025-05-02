import sys
from typing import Any, Literal

from pydantic import PositiveInt

from neurogym.config.base import ConfigBase


class MonitorPlotConfig(ConfigBase):
    """Plotting configuration during monitoring.

    Attributes:
        create: Whether to generate plots.
        title: Title to display on plots.
        ext: File extension for saved plot images.
        trigger: Unit used to trigger logging output.
        interval: Plotting interval in units of trigger.
    """

    create: bool = True
    title: str = ""
    ext: str = "png"
    trigger: Literal["trial", "step"] = "trial"
    interval: PositiveInt = 10


class MonitorLogConfig(ConfigBase):
    """Logging configuration during monitoring.

    Attributes:
        verbose: Whether to enable verbose output.
        format: Log format string.
        level: Logging level (e.g., INFO, DEBUG).
        trigger: Unit used to trigger logging output.
        interval: Logging interval in units of trigger.
    """

    verbose: bool = True
    format: str = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    level: str = "INFO"
    trigger: Literal["trial", "step"] = "trial"
    interval: PositiveInt = 10

    def make_config(self) -> dict[str, Any]:
        """Build logger configuration for Loguru."""
        return {
            "handlers": [
                {
                    "sink": sys.stderr,
                    "format": self.format,
                    "level": self.level,
                    "enqueue": True,
                    "diagnose": False,
                },
            ],
        }


class MonitorConfig(ConfigBase):
    """Top-level configuration for monitoring.

    Attributes:
        name: Optional monitor name.
        log: Logging configuration.
        plot: Plotting configuration.
    """

    name: str = ""
    log: MonitorLogConfig = MonitorLogConfig()
    plot: MonitorPlotConfig = MonitorPlotConfig()

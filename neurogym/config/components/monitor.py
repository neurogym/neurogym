import sys
from typing import Any, Literal

from neurogym.config.base import ConfigBase
from neurogym.config.components.trigger import TriggerConfig


class MonitorPlotConfig(TriggerConfig):
    """Plotting configuration during monitoring.

    Attributes:
        create: Whether to generate plots.
        title: Title to display on plots.
        ext: File extension for saved plot images.
        trigger: 'trials' or 'steps'.
        value: Plotting interval in units of trigger.
    """

    create: bool = True
    title: str = "NeuroGym"
    ext: str = "png"
    trigger: Literal["trials", "steps"] = "trials"
    value: int = 10


class MonitorLogConfig(TriggerConfig):
    """Logging configuration during monitoring.

    Attributes:
        verbose: Whether to enable verbose output.
        format: Log format string.
        level: Logging level (e.g., INFO, DEBUG).
        trigger: 'trials' or 'steps'.
        value: Logging interval in units of trigger.
    """

    verbose: bool = True
    format: str = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    level: str = "INFO"
    trigger: Literal["trials", "steps"] = "trials"
    value: int = 10

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

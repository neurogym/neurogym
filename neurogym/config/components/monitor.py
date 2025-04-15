import sys
from typing import Any, Literal

from neurogym.config.base import ConfigBase


class MonitorPlotConfig(ConfigBase):
    """Configuration options related to plotting as part of monitoring.

    create: A toggle to switch plotting on or off.
    title: The title to display in the plot.
    ext: Image extension.
    trigger: The metric used to trigger events such as plotting.
    interval: Plotting interval.
    """

    title: str = "NeuroGym"
    create: bool = True
    ext: str = "png"
    trigger: Literal["trial", "step"] = "trial"
    interval: int = 10


class MonitorLogConfig(ConfigBase):
    """Configuration options related to logging.

    verbose: A toggle indicating that Neurogym logging output should be more verbose.
    format: The logger output format (cf. the Loguru documentation at https://loguru.readthedocs.io/en/stable/index.html).
    level: The logging level (DEBUG, INFO, etc.).
    trigger: The metric used to trigger events such as plotting.
    interval: Logging interval (in units of 'trigger'; cf. :ref:`MonitorConfig.trigger`).
    """

    verbose: bool = True
    format: str = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    level: str = "INFO"
    trigger: Literal["trial", "step"] = "trial"
    interval: int = 10

    def make_config(self) -> dict[str, Any]:
        """A convenience method for constructing a logger configuration.

        Returns:
            A configuration dictionary.
        """
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
    """Configuration options related to monitoring.

    trigger: Subconfiguration The metric used to trigger events such as plotting.
    plot: Subconfiguration  to plotting (cf. :ref:`MonitorPlotConfig`)
    """

    log: MonitorLogConfig = MonitorLogConfig()
    plot: MonitorPlotConfig = MonitorPlotConfig()

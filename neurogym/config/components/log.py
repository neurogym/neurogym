import sys
from typing import Any

from neurogym.config.base import ConfBase


class LogConf(ConfBase):
    """Logger configuration."""

    verbose: bool = True
    format: str = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    level: str = "INFO"
    interval: int = 10

    def make_config(self) -> dict[str, Any]:
        return {
            "handlers": [
                {
                    "sink": sys.stdout,
                    "format": self.format,
                    "level": self.level,
                },
            ],
        }

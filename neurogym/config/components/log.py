from neurogym.config.base import ConfBase


class LogConf(ConfBase):
    """Logger configuration."""

    verbose: bool = True
    format: str = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"
    level: str = "INFO"
    # Logging interval in steps
    interval: int = 10

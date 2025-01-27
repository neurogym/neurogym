# --------------------------------------
from neurogym.config.base import ConfBase

class LogConf(ConfBase):
    '''
    Logger configuration.
    '''

    verbose: bool = True
    format: str = "Neurogym"
    level: str = "INFO"
    # Logging interval in steps
    interval: int = 100

from neurogym.config.base import ConfBase


class EnvConf(ConfBase):
    """Environment configuration."""

    name: str | None = None
    steps: int = 100000
    trials: int = 100
    dt: int = 10

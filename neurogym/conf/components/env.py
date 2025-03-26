from neurogym.conf.base import ConfBase


class EnvConf(ConfBase):
    """Configuration options related to environments.

    name: The name of the current environment.
    dt: Time interval between steps (in ms).
    """

    name: str | None = None
    dt: int = 10

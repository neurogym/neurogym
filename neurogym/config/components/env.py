from neurogym.config.base import ConfigBase


class EnvConfig(ConfigBase):
    """Configuration options related to environments.

    name: The name of the current environment.
    dt: Time interval between steps (in ms).
    """

    name: str | None = None
    dt: int = 10

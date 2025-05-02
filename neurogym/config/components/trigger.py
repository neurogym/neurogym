from typing import Literal

from pydantic import PositiveInt

from neurogym.config.base import ConfigBase


class TriggerConfig(ConfigBase):
    """Generic configuration for components that run by 'trials' or 'steps'.

    Attributes:
        trigger: Either 'trials' or 'steps'.
        value: Positive integer corresponding to the trigger.
    """

    trigger: Literal["trials", "steps"]
    value: PositiveInt

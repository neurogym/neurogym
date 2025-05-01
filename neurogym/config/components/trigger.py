from typing import Literal

from pydantic import model_validator

from neurogym.config.base import ConfigBase


class TriggerConfig(ConfigBase):
    """Generic configuration for components that run by 'trials' or 'steps'.

    Attributes:
        trigger: Either 'trials' or 'steps'.
        value: Positive integer corresponding to the trigger.
    """

    trigger: Literal["trials", "steps"]
    value: int

    @model_validator(mode="after")
    def validate_trigger(self):
        """Ensure value is positive and trigger is valid."""
        if self.value <= 0:
            msg = "Value must be a positive integer."
            raise ValueError(msg)
        if self.trigger not in ("trials", "steps"):
            msg = "Trigger must be either 'trials' or 'steps'"
            raise ValueError(msg)
        return self

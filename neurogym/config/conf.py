import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, TomlConfigSettingsSource

from neurogym.config.base import ConfBase
from neurogym.config.components.log import LogConf
from neurogym.config.components.monitor import MonitorConf
from neurogym.config.components.paths import CONFIG_DIR, PathConf


class Conf(ConfBase):
    """Main configuration."""

    paths: PathConf = PathConf()
    log: LogConf = LogConf()
    monitor: MonitorConf = MonitorConf()

    # A class method that tries to identify all
    # the possible configuration files to load.
    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[BaseSettings],
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, settings_cls.settings_file()),) #type: ignore[attr-defined]

    @classmethod
    def settings_file(cls):
        return CONFIG_DIR / "settings.toml"

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def capitalize(cls, value: str) -> str:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {
                k: (os.path.expandvars(v) if isinstance(v, str) else v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [(os.path.expandvars(v) if isinstance(v, str) else v) for v in value]
        return value


# Configuration object
# ==================================================
conf = Conf()

# Logger configuration
# ==================================================
log_config: dict[Any, Any] = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": conf.log.format,
            "level": conf.log.level,
        },
    ],
}

logger.configure(**log_config)

# Enable colour tags in messages.
logger = logger.opt(colors=True)
logger.info("Logger configured.")

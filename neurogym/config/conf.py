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
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Default configuration file
        config_file = (TomlConfigSettingsSource(settings_cls, CONFIG_DIR / "settings.toml"),)

        local_settings = getattr(cls, "settings_file", None)
        if local_settings is not None and Path(local_settings).exists():
            # Check if there is a configuration file
            # for this specific class.
            config_file = (TomlConfigSettingsSource(settings_cls, local_settings),)

        return config_file

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def capitalize(cls, value: str) -> str:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in value.items()}
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

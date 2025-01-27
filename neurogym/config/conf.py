# --------------------------------------
import os

# --------------------------------------
import sys

# --------------------------------------
from typing import Type

# --------------------------------------
from pathlib import Path

# --------------------------------------
from loguru import logger

# --------------------------------------
from pydantic import field_validator

# --------------------------------------
from pydantic_settings import BaseSettings
from pydantic_settings import TomlConfigSettingsSource
from pydantic_settings import PydanticBaseSettingsSource

# --------------------------------------
from neurogym.config.base import ConfBase
from neurogym.config.components.log import LogConf
from neurogym.config.components.paths import CONFIG_DIR
from neurogym.config.components.paths import PathConf
from neurogym.config.components.monitor import MonitorConf

class Conf(ConfBase):
    """
    Main configuration.
    """

    paths: PathConf = PathConf()
    log: LogConf = LogConf()
    monitor: MonitorConf = MonitorConf()

    # A class method that tries to identify all
    # the possible configuration files to load.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        # Default configuration file
        config_file = (
            TomlConfigSettingsSource(settings_cls, CONFIG_DIR / "settings.toml"),
        )

        local_settings = getattr(cls, "settings_file", None)
        if local_settings is not None and Path(local_settings).exists():
            # Check if there is a configuration file
            # for this specific class.
            config_file = TomlConfigSettingsSource(settings_cls, local_settings)

        return tuple(config_file)

    @field_validator('*', mode='before', check_fields=False)
    @classmethod
    def capitalize(cls, value: str) -> str:
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in value.items()}
        elif isinstance(value, list):
            return [(os.path.expandvars(v) if isinstance(v, str) else v) for v in value]
        return value


# Configuration object
# ==================================================
conf = Conf()

# Logger configuration
# ==================================================
log_config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": conf.log.format,
            "level": conf.log.level,
        }
    ]
}

logger.configure(**log_config)

# Enable colour tags in messages.
logger = logger.opt(colors=True)
logger.info(f"Logger configured.")

import os
from pathlib import Path

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from neurogym.config.base import CONF_DIR, LOCAL_DIR, PKG_DIR, ROOT_DIR, ConfBase
from neurogym.config.components.env import EnvConf
from neurogym.config.components.log import LogConf
from neurogym.config.components.monitor import MonitorConf


class Conf(ConfBase):
    """Main configuration."""

    # Configuration file
    conf_file: Path | None = Path.cwd() / "conf.toml"

    # Some useful paths.
    # Except for local_dir, the others cannot be modified.
    root_dir: Path = Field(frozen=True, default=ROOT_DIR)
    pkg_dir: Path = Field(frozen=True, default=PKG_DIR)
    conf_dir: Path = Field(frozen=True, default=CONF_DIR)
    local_dir: Path = LOCAL_DIR

    # Subconfiguration
    env: EnvConf = EnvConf()
    log: LogConf = LogConf()
    monitor: MonitorConf = MonitorConf()

    def __init__(
        self,
        conf_file: str | Path | None = None,
        *args,
        **kwargs,
    ):
        self.__class__.conf_file = None
        if isinstance(conf_file, (str | Path)):
            conf_file = Path(conf_file)
            if conf_file.exists():
                self.__class__.conf_file = Path(conf_file)

        super().__init__(*args, **kwargs)

    # A class method that tries to identify all
    # the possible configuration files to load.
    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[BaseSettings],
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, cls.conf_file),)  # type: ignore[attr-defined]

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def expand_env_vars(cls, value: str) -> str:
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
logger.remove()
logger.configure(**conf.log.make_config())

# Enable colour tags in messages.
logger = logger.opt(colors=True)

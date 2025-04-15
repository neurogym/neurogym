import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import (
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from neurogym.config.base import CONFIG_DIR, LOCAL_DIR, PKG_DIR, ROOT_DIR, ConfigBase
from neurogym.config.components.agent import AgentConfig
from neurogym.config.components.env import EnvConfig
from neurogym.config.components.monitor import MonitorConfig


class Config(ConfigBase):
    """Main configuration.

    NOTE: The `root_dir` `pkg_dir` and `config_dir` variables are provided only for convenience
    and cannot be modified. `local_dir` is the only one that can be assigned.

    config_file: A user-settable configuration file in TOML format.
    root_dir: The root directory of the repository.
    pkg_dir: The directory where the neurogym package is located.
    config_dir: The directory of the configuration module.
    local_dir: A local directory that should receive all kinds of program output (e.g., plots).
    agent: Options for agents (cf. :ref:`AgentConfig`).
    env: Options for environments (cf. :ref:`EnvConfig`).
    monitor: Subconfiguration for monitoring options (cf. :ref:`MonitorConfig`).
    """

    # TOML configuration file
    config_file: Path | None = Path.cwd() / "config.toml"

    root_dir: Path = Field(frozen=True, default=ROOT_DIR)
    pkg_dir: Path = Field(frozen=True, default=PKG_DIR)
    config_dir: Path = Field(frozen=True, default=CONFIG_DIR)
    local_dir: Path = LOCAL_DIR

    # Subconfiguration
    agent: AgentConfig = AgentConfig()
    env: EnvConfig = EnvConfig()
    monitor: MonitorConfig = MonitorConfig()

    def __init__(
        self,
        config_file: str | Path | None = None,
        *args,
        **kwargs,
    ):
        self.__class__.config_file = None
        if config_file is not None:
            if isinstance(config_file, str):
                config_file = Path(config_file)
            if config_file.exists():
                self.__class__.config_file = config_file

        super().__init__(*args, **kwargs)

    # A class method that tries to identify all
    # the possible configuration files to load.
    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[ConfigBase],
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, cls.config_file),)

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


# Default configuration object
# ==================================================
config = Config()

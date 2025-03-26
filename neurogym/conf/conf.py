import os
from pathlib import Path


from pydantic import Field, field_validator
from pydantic_settings import (
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from neurogym.conf.base import CONF_DIR, LOCAL_DIR, PKG_DIR, ROOT_DIR, ConfBase
from neurogym.conf.components.agent import AgentConf
from neurogym.conf.components.env import EnvConf
from neurogym.conf.components.monitor import MonitorConf


class Conf(ConfBase):
    """Main configuration.

    conf_file: A user-settable configuration file in TOML format.
    root_dir: The root directory of the repository.
    pkg_dir: The directory where the neurogym package is located.
    conf_dir: The directory of the configuration module.
    local_dir: A local directory that should receive all kinds of program output (e.g., plots).
    agent: Options for agents (cf. :ref:`AgentConf`).
    env: Options for environments (cf. :ref:`EnvConf`).
    monitor: Subconfiguration for monitoring options (cf. :ref:`MonitorConf`).
    """

    # TOML configuration file
    conf_file: Path | None = Path.cwd() / "conf.toml"

    root_dir: Path = Field(frozen=True, default=ROOT_DIR)
    pkg_dir: Path = Field(frozen=True, default=PKG_DIR)
    conf_dir: Path = Field(frozen=True, default=CONF_DIR)
    local_dir: Path = LOCAL_DIR

    # Subconfiguration
    agent: AgentConf = AgentConf()
    env: EnvConf = EnvConf()
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
        settings_cls: type[ConfBase],
        **kwargs,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, cls.conf_file),)

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def expand_env_vars(cls, value: str) -> str:
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


# Default donfiguration object
# ==================================================
conf = Conf()

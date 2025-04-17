from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent
PKG_DIR = ROOT_DIR / "neurogym"
CONFIG_DIR = PKG_DIR / "config"
LOCAL_DIR = ROOT_DIR / "local"


class ConfigBase(BaseSettings):
    """Configuration base class."""

    # This is a built-in attribute that is overriden here.
    model_config = SettingsConfigDict(
        env_prefix="NGYM_",
        env_nested_delimiter=".",
        case_sensitive=True,
        nested_model_default_partial_update=True,
        extra="allow",
    )

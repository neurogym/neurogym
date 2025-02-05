from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "neurogym"
LOCAL_DIR = ROOT_DIR / "local"
CONFIG_DIR = PACKAGE_DIR / "config"


class ConfBase(BaseSettings):
    """Configuration base class."""

    model_config = SettingsConfigDict(env_prefix="NGYM_", case_sensitive=True)

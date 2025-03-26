import enum
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent
PKG_DIR = ROOT_DIR / "neurogym"
CONF_DIR = PKG_DIR / "conf"
LOCAL_DIR = ROOT_DIR / "local"

# HACK: Revise this if the minimal Python version
# is bumped to >= 3.11
if sys.version_info[:2] <= (3, 10):

    class StrEnum(str, enum.Enum):
        pass

else:
    StrEnum = enum.StrEnum


class ConfBase(BaseSettings):
    """Configuration base class."""

    # This is a built-in attribute that is overriden here.
    model_config = SettingsConfigDict(
        env_prefix="NGYM_",
        case_sensitive=True,
        nested_model_default_partial_update=True,
        extra="allow",
    )

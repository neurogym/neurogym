from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent
PKG_DIR = ROOT_DIR / "neurogym"
CONFIG_DIR = PKG_DIR / "config"
LOCAL_DIR = ROOT_DIR / "local"


class ConfigBase(BaseSettings):
    """Base class for project configuration models.

    Extends Pydantic's `BaseSettings` to support loading from environment variables
    and other sources, with custom behavior controlled via `model_config`.

    Key `model_config` options:
    - `env_prefix`: Only environment variables with this prefix (e.g., "NGYM_") are used.
    - `env_nested_delimiter`: Enables nested fields (e.g., "NGYM_DB__USER").
    - `case_sensitive`: Enforces case-sensitive variable names.
    - `nested_model_default_partial_update`: Supports partial updates to nested models.
    - `extra="allow"`: Allows undeclared fields in the config (for flexibility).

    Note: These settings only affect environment variable loading, not file-based configs (e.g., TOML).
    """

    # Pydantic v2-style configuration for BaseSettings
    model_config = SettingsConfigDict(
        env_prefix="NGYM_",  # Look for environment variables starting with this
        env_nested_delimiter="__",  # Supports nested fields like NGYM_DB.USER
        case_sensitive=False,  # Makes variable names not case-sensitive
        nested_model_default_partial_update=True,  # Allows partial updates to nested models
        extra="allow",  # Accepts extra fields not defined in the model
        validate_assignment=True,  # Validates assignments to model fields
    )

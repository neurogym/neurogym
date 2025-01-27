from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

class ConfBase(BaseSettings):
    """
    Configuration base class.
    """

    model_config = SettingsConfigDict(env_prefix="NGYM_", case_sensitive=True)

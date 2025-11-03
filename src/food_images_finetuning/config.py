from typing import ClassVar

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HFSettings(BaseModel):
    token: SecretStr = Field(default=SecretStr(""), description="Hugging Face API token")


# -----------------------------
# Main Settings
# -----------------------------
class Settings(BaseSettings):
    hf_settings: HFSettings = HFSettings()

    # Pydantic v2 model config
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )


# -----------------------------
# Instantiate settings
# -----------------------------
settings = Settings()

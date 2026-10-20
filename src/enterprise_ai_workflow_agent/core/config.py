from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_ALLOWED_ROUNDS = 8
MAX_QUESTION_LENGTH = 4000


class Settings(BaseSettings):
    app_name: str = "Enterprise AI Workflow Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./opspilot.db"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = Field(default=0.2, ge=0, le=2)
    ollama_num_predict: int = Field(default=512, ge=1)
    ollama_num_ctx: int = Field(default=4096, ge=512)
    ollama_preflight_timeout_seconds: float = Field(default=3.0, gt=0)

    # The timeout covers the whole request, including waiting for a free slot.
    investigation_timeout_seconds: float = Field(default=300.0, gt=0)
    investigation_max_rounds: int = Field(default=4, ge=1, le=MAX_ALLOWED_ROUNDS)
    investigation_max_concurrency: int = Field(default=1, ge=1)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

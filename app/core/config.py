"""Core application configuration supporting pydantic-settings with fallback."""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field

load_dotenv()

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class _BaseSettingsConfig(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
except ImportError:
    from pydantic import BaseModel

    class _BaseSettingsConfig(BaseModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            env_vals = {}
            fields = getattr(self.__class__, "model_fields", {})
            for field_name, field_info in fields.items():
                alias = field_info.alias or field_name.upper()
                if alias in os.environ:
                    val = os.environ[alias]
                    target_type = field_info.annotation
                    if target_type is int or target_type == Optional[int]:
                        try:
                            env_vals[field_name] = int(val)
                        except ValueError:
                            pass
                    elif target_type is float or target_type == Optional[float]:
                        try:
                            env_vals[field_name] = float(val)
                        except ValueError:
                            pass
                    elif target_type is bool:
                        env_vals[field_name] = val.lower() in ("true", "1", "yes")
                    else:
                        env_vals[field_name] = val
            env_vals.update(kwargs)
            super().__init__(**env_vals)


class Settings(_BaseSettingsConfig):
    """Central configuration for RelayCore system."""

    # Core Application
    app_name: str = Field(default="RelayCore", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    version: str = "0.1.0"

    # Groq API Configuration
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    # Upstream Provider Guardrails (Groq Free Tier limits)
    groq_rpm_limit: int = Field(default=30, alias="GROQ_RPM_LIMIT")
    groq_tpm_limit: int = Field(default=6000, alias="GROQ_TPM_LIMIT")
    groq_rpd_limit: int = Field(default=14400, alias="GROQ_RPD_LIMIT")

    # Stand-in Simulator Settings
    simulator_enabled: bool = Field(default=False, alias="SIMULATOR_ENABLED")
    simulator_port: int = Field(default=8001, alias="SIMULATOR_PORT")
    simulator_default_ttft_ms: int = Field(default=320, alias="SIMULATOR_DEFAULT_TTFT_MS")
    simulator_failure_rate_500: float = Field(default=0.02, alias="SIMULATOR_FAILURE_RATE_500")
    simulator_failure_rate_503: float = Field(default=0.01, alias="SIMULATOR_FAILURE_RATE_503")

    # Queue & Concurrency Management
    max_queue_depth: int = Field(default=100, alias="MAX_QUEUE_DEPTH")
    worker_concurrency: int = Field(default=5, alias="WORKER_CONCURRENCY")
    default_client_timeout_ms: int = Field(default=15000, alias="DEFAULT_CLIENT_TIMEOUT_MS")
    max_retries: int = Field(default=2, alias="MAX_RETRIES")

    # Retrieval and Storage
    data_dir: str = Field(default="data", alias="DATA_DIR")
    results_dir: str = Field(default="results", alias="RESULTS_DIR")
    default_chunk_size_words: int = Field(default=350, alias="DEFAULT_CHUNK_SIZE_WORDS")
    default_chunk_overlap_words: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP_WORDS")
    default_top_k: int = Field(default=3, alias="DEFAULT_TOP_K")


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings singleton."""
    return Settings()

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything addressable comes from the environment — no hardcoded hosts."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = "redis://:gavno@127.0.0.1:6379/0"
    index_dir: Path = Path("index_data")

    embedding_dim: int = 384
    default_top_k: int = 200

    # 0 disables the background watcher; the index then moves only on POST /reload.
    reload_watch_seconds: float = 60.0

    host: str = "0.0.0.0"
    port: int = 8001

    # How many index versions the builder keeps on disk before pruning.
    keep_versions: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    BENTO_HOME: Path = Path(__file__).resolve().parents[1] / "bento_store"


settings = Settings()

import torch
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    GPU_PER_MODEL: float = 0.25
    CPU_PER_MODEL: float = 1.0
    DEVICE: torch.device = torch.device("cuda")

    @validator("DEVICE", pre=True)
    def validate(cls, val):
        return torch.device(val)


settings = Settings()

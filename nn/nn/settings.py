from pathlib import Path

from pydantic import BaseSettings

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights"


class YoloSettings(BaseSettings):
    CONFIDENCE: float = 0.57
    WEIGHTS: Path = WEIGHTS_PATH / "yolo.pth"


class LprnetSettings(BaseSettings):
    NUM_CLASSES: int = 23
    OUT_INDICES: list[int] = [2, 6, 13, 22]
    WEIGHTS: Path = WEIGHTS_PATH / "lprnet.pth"


class StnSettings(BaseSettings):
    WEIGHTS: Path = WEIGHTS_PATH / "stn.pth"


class ModelSettings(BaseSettings):
    YOLO: YoloSettings = YoloSettings()
    LPRNET: LprnetSettings = LprnetSettings()
    STN: StnSettings = StnSettings()


model_settings = ModelSettings()

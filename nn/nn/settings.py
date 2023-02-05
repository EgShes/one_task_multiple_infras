from pathlib import Path

from pydantic import BaseSettings

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights"


class VocabularySettings(BaseSettings):
    VOCAB: list[str] = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "E",
        "K",
        "M",
        "H",
        "O",
        "P",
        "C",
        "T",
        "Y",
        "X",
        "-",
    ]


class YoloSettings(BaseSettings):
    CONFIDENCE: float = 0.57
    PREDICT_SIZE: int = 1280
    WEIGHTS: Path = WEIGHTS_PATH / "yolo.pt"


class LprnetSettings(BaseSettings):
    NUM_CLASSES: int = 23
    OUT_INDICES: list[int] = [2, 6, 13, 22]
    IMG_SIZE: tuple[int, int] = (94, 24)
    WEIGHTS: Path = WEIGHTS_PATH / "lprnet.pth"


class StnSettings(BaseSettings):
    WEIGHTS: Path = WEIGHTS_PATH / "stn.pth"


class Settings(BaseSettings):
    YOLO: YoloSettings = YoloSettings()
    LPRNET: LprnetSettings = LprnetSettings()
    STN: StnSettings = StnSettings()
    VOCAB: VocabularySettings = VocabularySettings()


settings = Settings()

import torch

from nn.models.yolo import load_yolo
from nn.settings import model_settings


def test_load():
    load_yolo(
        weights=model_settings.YOLO.WEIGHTS,
        confidence=model_settings.YOLO.CONFIDENCE,
        device=torch.device("cpu"),
    )

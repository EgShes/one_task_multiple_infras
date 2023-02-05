import torch

from nn.models.yolo import load_yolo
from nn.settings import settings


def test_load():
    load_yolo(
        weights=settings.YOLO.WEIGHTS,
        confidence=settings.YOLO.CONFIDENCE,
        device=torch.device("cpu"),
    )

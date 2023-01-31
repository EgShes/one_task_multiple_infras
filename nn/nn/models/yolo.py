from pathlib import Path

import torch


def load_model(weights: Path, confidence: float, device: torch.device):
    model = (
        torch.hub.load("ultralytics/yolov5", "custom", path=weights).eval().to(device)
    )
    model.conf = confidence
    return model

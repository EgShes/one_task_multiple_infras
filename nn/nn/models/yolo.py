from pathlib import Path

import torch


def load_yolo(
    weights: Path, confidence: float, device: torch.device
) -> torch.nn.Module:
    model = (
        torch.hub.load("ultralytics/yolov5", "custom", path=weights).eval().to(device)
    )
    model.conf = confidence
    return model

from pathlib import Path

import torch

from nn.models import load_yolo
from nn.settings import settings

if __name__ == "__main__":
    model_path = (
        Path(__file__).resolve().parents[1]
        / "model_repository"
        / "yolo"
        / "1"
        / "model.pt"
    )
    model_path.parent.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    dummy_input = torch.randn(1, 3, 768, 1280, device=device)
    model = load_yolo(
        settings.YOLO.WEIGHTS,
        settings.YOLO.CONFIDENCE,
        device,
    )

    traced_model = torch.jit.trace(model, dummy_input, strict=False)
    traced_model.save(model_path)

    with torch.no_grad():
        orig_res = model(dummy_input)
        traced_res = model(dummy_input)

    assert torch.all(orig_res == traced_res)

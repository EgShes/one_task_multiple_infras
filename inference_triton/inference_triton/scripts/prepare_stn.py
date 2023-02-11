from pathlib import Path

import torch

from nn.models import load_stn
from nn.settings import settings

if __name__ == "__main__":
    model_path = (
        Path(__file__).resolve().parents[1]
        / "model_repository"
        / "stn"
        / "1"
        / "model.pt"
    )
    model_path.parent.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    dummy_input = torch.randn(1, 3, 24, 94, device=device)
    model = load_stn(settings.STN.WEIGHTS, device)

    traced_model = torch.jit.trace(model, [dummy_input])
    traced_model.save(model_path)

    with torch.no_grad():
        orig_res = model(dummy_input)
        traced_res = model(dummy_input)

    assert torch.all(orig_res == traced_res)

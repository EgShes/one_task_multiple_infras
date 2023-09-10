from pathlib import Path

import torch

from nn.models import load_stn


class StnModel:
    def __init__(self, model_file: Path, device: torch.device = torch.device("cpu")):
        self.model = load_stn(model_file, device)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions

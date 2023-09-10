from pathlib import Path

import torch

from nn.models import load_lprnet
from nn.settings import settings as settings_nn


class LprnetModel:
    def __init__(self, model_file: Path, device: torch.device = torch.device("cpu")):
        self.model = load_lprnet(
            model_file,
            settings_nn.LPRNET.NUM_CLASSES,
            settings_nn.LPRNET.OUT_INDICES,
            device,
        )

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions

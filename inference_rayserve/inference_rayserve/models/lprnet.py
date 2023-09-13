from pathlib import Path

import torch
from ray import serve

from inference_rayserve.settings import settings
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
        self.device = device

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions


LprnetDeployment = serve.deployment(
    LprnetModel,
    "lprnet",
    ray_actor_options={
        "num_cpus": settings.CPU_PRE_MODEL,
        "num_gpus": settings.GPU_PER_MODEL,
    },
)

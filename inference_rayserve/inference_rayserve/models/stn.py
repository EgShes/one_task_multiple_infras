from pathlib import Path

import torch
from ray import serve

from inference_rayserve.settings import settings
from nn.models import load_stn


class StnModel:
    def __init__(self, model_file: Path, device: torch.device = torch.device("cpu")):
        self.model = load_stn(model_file, device)
        self.device = device

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions


StnDeployment = serve.deployment(
    StnModel,
    "stn",
    ray_actor_options={
        "num_cpus": settings.CPU_PRE_MODEL,
        "num_gpus": settings.GPU_PER_MODEL,
    },
)

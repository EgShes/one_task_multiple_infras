import bentoml
import torch
import numpy as np

from nn.models import load_yolo
from nn.settings import settings


class YoloRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model_file):
        device = torch.device("cuda")
        self.model = load_yolo(
            model_file,
            settings.YOLO.CONFIDENCE,
            device,
        )

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def predict(self, inputs: np.ndarray):
        return self.model(inputs)

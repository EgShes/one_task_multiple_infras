from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from ray import serve
from starlette.requests import Request

from inference_rayserve.settings import settings
from nn.inference.predictor import prepare_detection_input, prepare_recognition_input
from nn.models import load_yolo
from nn.settings import settings as settings_nn


class YoloPrediction(NamedTuple):
    plates: np.ndarray
    coordinates: np.ndarray


class YoloModel:
    def __init__(self, model_file: Path, device: torch.device = torch.device("cpu")):
        self.model = load_yolo(model_file, settings_nn.YOLO.CONFIDENCE, device)

    def predict(self, inputs: np.ndarray) -> YoloPrediction:
        inputs = inputs.astype(np.float32)
        image = prepare_detection_input(inputs)

        detection = self.model(image, size=settings_nn.YOLO.PREDICT_SIZE)

        df_results = detection.pandas().xyxy[0]
        plates = prepare_recognition_input(
            df_results, image, return_torch=False
        ).astype(
            np.float32
        )  # (n, 3, h, w)
        coordinates = df_results[["xmin", "ymin", "xmax", "ymax"]].to_numpy()  # (n, 4)

        return YoloPrediction(plates, coordinates)

    async def __call__(self, http_request: Request) -> YoloPrediction:
        image = np.array(await http_request.json())
        print(image.shape, flush=True)
        prediction = self.predict(image)
        return prediction


YoloDeployment = serve.deployment(
    YoloModel,
    "yolo",
    ray_actor_options={
        "num_cpus": settings.CPU_PER_MODEL,
        "num_gpus": settings.GPU_PER_MODEL,
    },
)

from typing import NamedTuple

import bentoml
import numpy as np
import torch

from nn.inference.predictor import prepare_detection_input, prepare_recognition_input
from nn.models import load_yolo
from nn.settings import settings


class YoloPrediction(NamedTuple):
    plates: np.ndarray
    coordinates: np.ndarray


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
        image = prepare_detection_input(inputs)

        detection = self.model(image, size=settings.YOLO.PREDICT_SIZE)

        df_results = detection.pandas().xyxy[0]
        plates = prepare_recognition_input(
            df_results, image, return_torch=False
        ).astype(
            np.float32
        )  # (n, 3, h, w)
        coordinates = df_results[["xmin", "ymin", "xmax", "ymax"]].to_numpy()  # (n, 4)

        return YoloPrediction(plates, coordinates)

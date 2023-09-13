from typing import List, NamedTuple

import numpy as np
import torch
from pydantic import BaseModel
from ray import serve
from starlette.requests import Request

from inference_rayserve.settings import settings
from nn.inference.decode import beam_decode
from nn.inference.predictor import filter_predictions
from nn.settings import settings as settings_nn


class PlateCoordinates(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float


class PlatePrediction(BaseModel):
    coordinates: List[PlateCoordinates]
    texts: List[str]


@serve.deployment(
    "plate_recognition",
    ray_actor_options={
        "num_cpus": settings.CPU_PRE_MODEL,
        "num_gpus": settings.GPU_PER_MODEL,
    },
)
class PlateRecognitionDeployment:
    def __init__(self, yolo, stn, lprnet):
        self.yolo = yolo
        self.stn = stn
        self.lprnet = lprnet

    async def __call__(self, http_request: Request) -> PlatePrediction:
        image = np.array(await http_request.json())

        ref = await self.yolo.predict.remote(image)
        plates, coordinates = await ref

        num_plates = plates.shape[0]
        if num_plates == 0:
            return PlatePrediction(coordinates=[], texts=[])

        plates = torch.from_numpy(plates)
        ref = await self.stn.predict.remote(plates)
        plate_features = await ref

        ref = await self.lprnet.predict.remote(plate_features)
        text_features = await ref

        # postprocess texts
        labels, log_likelihood, _ = beam_decode(
            text_features.cpu().numpy(),
            settings_nn.VOCAB.VOCAB,
        )
        predicted_plates = filter_predictions(labels, log_likelihood)
        predicted_plates = [
            text or "" for text in predicted_plates
        ]  # replace Nones with empty string

        coordinates = [PlateCoordinates(c[0], c[1], c[2], c[3]) for c in coordinates]

        return PlatePrediction(coordinates=coordinates, texts=predicted_plates)

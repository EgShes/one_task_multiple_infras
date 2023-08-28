from typing import List

import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

from inference_bentoml.plate_recognition import (
    PlateCoordinates,
)
from inference_bentoml.plate_recognition import recognize as recognize_fn
from inference_bentoml.runners.yolo import YoloRunnable
from nn.settings import settings


class PlatePrediction(BaseModel):
    coordinates: List[PlateCoordinates]
    texts: List[str]


yolo_runner = bentoml.Runner(
    YoloRunnable,
    name="yolo",
    runnable_init_params={
        "model_file": settings.YOLO.WEIGHTS,
    },
)
stn_runner = bentoml.pytorch.get("stn:latest").to_runner()
lprnet_runner = bentoml.pytorch.get("lprnet:latest").to_runner()


svc = bentoml.Service(
    "plate_recognition", runners=[yolo_runner, stn_runner, lprnet_runner]
)


@svc.api(input=NumpyNdarray(), output=JSON(pydantic_model=PlatePrediction))
def recognize(inputs: np.ndarray) -> PlatePrediction:
    prediction = recognize_fn(inputs, yolo_runner, stn_runner, lprnet_runner)
    return PlatePrediction(coordinates=prediction.coordinates, texts=prediction.texts)

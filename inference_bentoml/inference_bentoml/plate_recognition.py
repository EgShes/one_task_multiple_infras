from typing import List, NamedTuple

import numpy as np
from bentoml import Runnable

from inference_bentoml.runners.yolo import YoloRunnable
from nn.inference.decode import beam_decode
from nn.inference.predictor import filter_predictions
from nn.settings import settings


class PlateCoordinates(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float


class PlatePrediction(NamedTuple):
    coordinates: List[PlateCoordinates]
    texts: List[str]


def recognize(
    inputs: np.ndarray,
    yolo_runner: YoloRunnable,
    stn_runner: Runnable,
    lprnet_runner: Runnable,
) -> PlatePrediction:
    plates, coordinates = yolo_runner.run(inputs)

    num_plates = plates.shape[0]
    if num_plates == 0:
        return PlatePrediction([], [])

    plate_features = stn_runner.run(plates)
    text_features = lprnet_runner.run(plate_features)

    # postprocess texts
    labels, log_likelihood, _ = beam_decode(
        text_features.cpu().numpy(),
        settings.VOCAB.VOCAB,
    )
    predicted_plates = filter_predictions(labels, log_likelihood)
    predicted_plates = [
        text or "" for text in predicted_plates
    ]  # replace Nones with empty string

    coordinates = [PlateCoordinates(c[0], c[1], c[2], c[3]) for c in coordinates]

    return PlatePrediction(coordinates, predicted_plates)

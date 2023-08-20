from typing import List

import bentoml
import cv2
import numpy as np
import pytest
import torch

from inference_bentoml.plate_recognition import recognize
from inference_bentoml.runners.yolo import YoloRunnable
from nn.settings import settings as settings


def test_yolo_model():
    inputs = np.ones((500, 800, 3))

    runner = bentoml.Runner(
        YoloRunnable,
        name="yolo",
        runnable_init_params={
            "model_file": settings.YOLO.WEIGHTS,
        },
    )
    runner.init_local()
    prediction = runner.run(inputs)

    assert prediction.plates.shape == (0,)
    assert prediction.plates.dtype == np.float32
    assert prediction.coordinates.shape == (0, 4)
    assert prediction.coordinates.dtype == np.object_


def test_stn_model():
    inputs = torch.randn(4, 3, 24, 94)

    runner = bentoml.pytorch.get("stn").to_runner()
    runner.init_local()
    prediction = runner.run(inputs)

    assert prediction.shape == inputs.shape
    assert prediction.dtype == torch.float32


def test_lprnet_model():
    inputs = torch.randn(4, 3, 24, 94)

    runner = bentoml.pytorch.get("lprnet").to_runner()
    runner.init_local()
    prediction = runner.run(inputs)

    assert prediction.shape == (4, 23, 18)
    assert prediction.dtype == torch.float32


@pytest.mark.parametrize(
    "img_path,expected_coordinates,expected_texts",
    [
        [
            "tests/data/car.jpg",
            [
                (232.44186, 814.19446, 324.64374, 841.9125),
                (1097.4425, 661.41547, 1141.9923, 674.0388),
                (1520.4563, 639.6942, 1567.8191, 653.18317),
                (1286.0554, 636.13745, 1317.4097, 645.0998),
            ],
            ["B840CK197", "", "", ""],
        ],
        ["tests/data/cat.jpeg", [], []],
    ],
)
def test_recognize(
    img_path: str,
    expected_coordinates: List[List[float]],
    expected_texts: List[str],
):
    inputs = cv2.imread(str(img_path))

    yolo_runner = bentoml.Runner(
        YoloRunnable,
        name="yolo",
        runnable_init_params={
            "model_file": settings.YOLO.WEIGHTS,
        },
    )
    yolo_runner.init_local()

    stn_runner = bentoml.pytorch.get("stn").to_runner()
    stn_runner.init_local()

    lprnet_runner = bentoml.pytorch.get("lprnet").to_runner()
    lprnet_runner.init_local()

    prediction = recognize(inputs, yolo_runner, stn_runner, lprnet_runner)

    assert prediction.texts == expected_texts
    for actual_coord, expected_coord in zip(
        prediction.coordinates, expected_coordinates
    ):
        assert all(
            [np.allclose(ac, ex) for ac, ex in zip(actual_coord, expected_coord)]
        )

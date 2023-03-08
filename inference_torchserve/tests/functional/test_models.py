from pathlib import Path
from typing import List

import numpy as np
import pytest
import requests

from inference_torchserve.data_models import DetectorPrediction, PlatePrediction


@pytest.mark.parametrize(
    "image,expected_plates",
    [
        (
            Path("tests/data/car.jpg"),
            [
                PlatePrediction(
                    xmin=232, ymin=813, xmax=324, ymax=842, confidence=0.929
                ),
                PlatePrediction(
                    xmin=1097, ymin=661, xmax=1142, ymax=674, confidence=0.892
                ),
                PlatePrediction(
                    xmin=1521, ymin=640, xmax=1566, ymax=652, confidence=0.861
                ),
                PlatePrediction(
                    xmin=1286, ymin=635, xmax=1316, ymax=644, confidence=0.689
                ),
            ],
        ),
        (Path("tests/data/cat.jpeg"), []),
    ],
)
def test_yolo(image: Path, expected_plates: List[PlatePrediction]):
    response = requests.post(
        "http://localhost:8080/predictions/yolo", data=image.open("rb").read()
    )
    prediction = DetectorPrediction.parse_obj(response.json())
    for pred_plate, exp_plate in zip(prediction.plates, expected_plates):
        assert pred_plate.xmin == exp_plate.xmin
        assert pred_plate.ymin == exp_plate.ymin
        assert pred_plate.xmax == exp_plate.xmax
        assert pred_plate.ymax == exp_plate.ymax
        assert pred_plate.confidence == pytest.approx(exp_plate.confidence, abs=1e-3)


def test_lprnet():
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 23, 18)

    inputs = np.random.randn(*input_shape).astype(np.float32).tobytes()

    response = requests.post("http://localhost:8080/predictions/lprnet", data=inputs)

    output = np.array(response.json()["data"])

    assert output.shape == output_shape

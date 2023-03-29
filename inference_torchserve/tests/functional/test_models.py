from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import requests

from inference_torchserve.data_models import PlatePrediction


@pytest.mark.parametrize(
    "image,expected_plates,expected_shape",
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
            (4, 3, 24, 94),
        ),
        (Path("tests/data/cat.jpeg"), [], (0,)),
    ],
)
def test_yolo(
    image: Path, expected_plates: List[PlatePrediction], expected_shape: Tuple[int]
):
    response = requests.post(
        "http://localhost:8080/predictions/yolo", data=image.open("rb").read()
    )
    prediction = response.json()
    for pred_plate, exp_plate in zip(prediction["coordinates"], expected_plates):
        pred_plate = PlatePrediction.parse_obj(pred_plate)
        assert pred_plate.xmin == exp_plate.xmin
        assert pred_plate.ymin == exp_plate.ymin
        assert pred_plate.xmax == exp_plate.xmax
        assert pred_plate.ymax == exp_plate.ymax
        assert pred_plate.confidence == pytest.approx(exp_plate.confidence, abs=1e-3)

    assert np.array(prediction["data"]).shape == expected_shape


def test_stn():
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 3, 24, 94)

    inputs = np.random.randn(*input_shape).astype(np.float32).tobytes()

    response = requests.post("http://localhost:8080/predictions/stn", data=inputs)

    output = np.array(response.json()["data"])

    assert output.shape == output_shape


def test_lprnet():
    input_shape = (4, 3, 24, 94)

    inputs = np.random.randn(*input_shape).astype(np.float32).tobytes()

    response = requests.post("http://localhost:8080/predictions/lprnet", data=inputs)

    texts = response.json()["data"]
    for text in texts:
        assert isinstance(text, str)

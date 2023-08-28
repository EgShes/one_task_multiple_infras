from typing import List

import cv2
import numpy as np
import pytest
from bentoml.client import Client


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

    client = Client.from_url("http://localhost:3000")

    prediction = client.call("recognize", inputs)

    assert prediction["texts"] == expected_texts
    for actual_coord, expected_coord in zip(
        prediction["coordinates"], expected_coordinates
    ):
        assert all(
            [
                np.allclose(ac, ex, atol=0.999)
                for ac, ex in zip(actual_coord, expected_coord)
            ]
        )

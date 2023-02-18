from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import tritonclient.http as httpclient


def send_request(
    client, model_name: str, input_shape: Tuple[int, ...], input_type: str
) -> np.ndarray:
    inputs, outputs = [], []

    inputs.append(httpclient.InferInput("input__0", input_shape, input_type))
    input_dtype = np.float32 if input_type == "FP32" else np.uint8
    inputs[0].set_data_from_numpy(np.random.randn(*input_shape).astype(input_dtype))

    outputs.append(httpclient.InferRequestedOutput("output__0", binary_data=False))
    results = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )
    return results.as_numpy("output__0")


@pytest.fixture(scope="session")
def triton_client() -> httpclient.InferenceServerClient:
    return httpclient.InferenceServerClient(url="localhost:8000")


def test_stn(triton_client):
    model_name = "stn"
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 3, 24, 94)
    input_type = "FP32"

    prediction = send_request(triton_client, model_name, input_shape, input_type)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32


def test_lprnet(triton_client):
    model_name = "lprnet"
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 23, 18)
    input_type = "FP32"

    prediction = send_request(triton_client, model_name, input_shape, input_type)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32


def test_yolo(triton_client):
    model_name = "yolo"
    input_shape = (500, 800, 3)
    output_shape = (0,)  # no detections
    input_type = "UINT8"

    prediction = send_request(triton_client, model_name, input_shape, input_type)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32


@pytest.mark.parametrize(
    "img_path,expected_coordinates,expected_texts",
    [
        [
            "tests/data/car.jpg",
            np.array(
                [
                    [232.44186, 814.19446, 324.64374, 841.9125],
                    [1097.4425, 661.41547, 1141.9923, 674.0388],
                    [1520.4563, 639.6942, 1567.8191, 653.18317],
                    [1286.0554, 636.13745, 1317.4097, 645.0998],
                ]
            ),
            np.array(["B840CK197", "", "", ""]),
        ],
        ["tests/data/cat.jpeg", np.empty((0, 4)), np.empty((0, 1))],
    ],
)
def test_plate_recognition(
    triton_client,
    img_path: Path,
    expected_coordinates: List[float],
    expected_texts: List[str],
):
    model_name = "plate_recognition"
    image = cv2.imread(str(img_path))

    inputs, outputs = [], []

    inputs.append(httpclient.InferInput("input__0", image.shape, "UINT8"))
    inputs[0].set_data_from_numpy(image)

    outputs.append(httpclient.InferRequestedOutput("coordinates", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("texts", binary_data=False))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )

    coordinates = results.as_numpy("coordinates")
    texts = results.as_numpy("texts")

    assert coordinates.shape == expected_coordinates.shape
    assert texts.shape == texts.shape

    assert np.allclose(coordinates, expected_coordinates)
    assert np.all(texts == expected_texts)

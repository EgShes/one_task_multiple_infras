from typing import Tuple

import numpy as np
import pytest
import tritonclient.http as httpclient


def send_request(
    client,
    model_name: str,
    input_shape: Tuple[int, ...],
) -> np.ndarray:
    inputs, outputs = [], []

    inputs.append(httpclient.InferInput("input__0", input_shape, "FP32"))
    inputs[0].set_data_from_numpy(np.random.randn(*input_shape).astype(np.float32))

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

    prediction = send_request(triton_client, model_name, input_shape)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32


def test_lprnet(triton_client):
    model_name = "lprnet"
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 23, 18)

    prediction = send_request(triton_client, model_name, input_shape)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32


def test_yolo(triton_client):
    model_name = "yolo"
    input_shape = (500, 800, 3)
    output_shape = (0,)  # no detections

    prediction = send_request(triton_client, model_name, input_shape)
    assert prediction.shape == output_shape
    assert prediction.dtype == np.float32

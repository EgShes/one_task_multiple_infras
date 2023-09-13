import numpy as np
import pytest
import torch

from inference_rayserve.models import LprnetModel, StnModel, YoloModel
from nn.settings import settings as settings_nn


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_yolo_model(device: torch.device):
    inputs = np.ones((500, 800, 3))

    model = YoloModel(settings_nn.YOLO.WEIGHTS, device)
    prediction = model.predict(inputs)

    assert prediction.plates.shape == (0,)
    assert prediction.plates.dtype == np.float32
    assert prediction.coordinates.shape == (0, 4)
    assert prediction.coordinates.dtype == np.object_


def test_stn_model(device: torch.device):
    inputs = torch.randn(4, 3, 24, 94).to(device)

    model = StnModel(settings_nn.STN.WEIGHTS, device)
    prediction = model.predict(inputs)

    assert prediction.shape == inputs.shape
    assert prediction.dtype == torch.float32


def test_lprnet_model(device: torch.device):
    inputs = torch.randn(4, 3, 24, 94).to(device)

    model = LprnetModel(settings_nn.LPRNET.WEIGHTS, device)
    prediction = model.predict(inputs)

    assert prediction.shape == (4, 23, 18)
    assert prediction.dtype == torch.float32

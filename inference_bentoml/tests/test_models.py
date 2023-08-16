import bentoml
import numpy as np
import torch

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

    assert np.array_equal(
        prediction.pandas().xyxy[0].to_numpy(), np.empty((0, 7), dtype=np.object_)
    )


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

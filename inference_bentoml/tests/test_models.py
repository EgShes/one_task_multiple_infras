import bentoml
import numpy as np

from inference_bentoml.runners.yolo import YoloRunnable
from nn.settings import settings


def test_yolo_model():
    runner = bentoml.Runner(
        YoloRunnable,
        name="yolo",
        runnable_init_params={
            "model_file": settings.YOLO.WEIGHTS,
        },
    )
    runner.init_local()
    inputs = np.ones((500, 800, 3))
    prediction = runner.run(inputs)

    assert np.array_equal(
        prediction.pandas().xyxy[0].to_numpy(), np.empty((0, 7), dtype=np.object_)
    )

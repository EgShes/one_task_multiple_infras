import numpy as np

from inference_rayserve.models.yolo import YoloModel
from nn.settings import settings as settings


def test_yolo_model():
    inputs = np.ones((500, 800, 3))

    model = YoloModel(settings.YOLO.WEIGHTS)
    prediction = model.predict(inputs)

    assert prediction.plates.shape == (0,)
    assert prediction.plates.dtype == np.float32
    assert prediction.coordinates.shape == (0, 4)
    assert prediction.coordinates.dtype == np.object_

from pathlib import Path

import torch

from nn.inference.decode import beam_decode
from nn.inference.predictor import Predictor
from nn.models import load_lprnet, load_stn, load_yolo
from nn.settings import settings


def test_predictor():
    device = torch.device("cpu")

    yolo = load_yolo(
        weights=settings.YOLO.WEIGHTS,
        confidence=settings.YOLO.CONFIDENCE,
        device=device,
    )
    lprnet = load_lprnet(
        weights=settings.LPRNET.WEIGHTS,
        num_classes=settings.LPRNET.NUM_CLASSES,
        out_indices=settings.LPRNET.OUT_INDICES,
        device=device,
    )
    stn = load_stn(weights=settings.STN.WEIGHTS, device=device)

    predictor = Predictor(
        yolo=yolo, lprn=lprnet, stn=stn, device=device, decode_fn=beam_decode
    )

    image_path = Path(__file__).resolve().parents[1] / "data" / "car.jpg"
    results = predictor.predict(image_path)

    assert len(results) == 4

    expected_numbers = ["B840CK197", None, None, None]
    for result, expected_number in zip(results, expected_numbers):
        assert result.number == expected_number

    for result in results:
        assert isinstance(result.number, (str, type(None)))
        assert isinstance(result.x_min, float)
        assert isinstance(result.y_min, float)
        assert isinstance(result.x_max, float)
        assert isinstance(result.y_max, float)

        assert result.x_min < result.x_max
        assert result.y_min < result.y_max

import torch

from nn.models.stn import SpatialTransformer, load_stn
from nn.settings import settings


def test_forward():
    bs, in_channels = 4, 3
    model = SpatialTransformer(in_channels=in_channels).eval()
    inputs = torch.Tensor(bs, 3, 24, 94)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == inputs.shape


def test_load():
    load_stn(weights=settings.STN.WEIGHTS, device=torch.device("cpu"))

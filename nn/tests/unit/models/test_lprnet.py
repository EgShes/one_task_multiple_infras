import torch

from nn.models.lprnet import LPRNet


def test_forward():
    bs, num_classes = 3, 23
    model = LPRNet(class_num=num_classes, out_indices=(2, 6, 13, 22)).eval()
    inputs = torch.Tensor(bs, 3, 24, 94)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape[0] == bs
    assert outputs.shape[1] == num_classes

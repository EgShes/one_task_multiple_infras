import os

import bentoml
import torch

from inference_bentoml.settings import settings as settings_ib
from nn.models import load_lprnet
from nn.settings import settings as settings_nn

os.environ["BENTOML_HOME"] = str(settings_ib.BENTO_HOME)


if __name__ == "__main__":
    device = torch.device("cuda")

    model = load_lprnet(
        settings_nn.LPRNET.WEIGHTS,
        settings_nn.LPRNET.NUM_CLASSES,
        settings_nn.LPRNET.OUT_INDICES,
        device,
    )

    bentoml.pytorch.save_model(
        "lprnet", model, signatures={"__call__": {"batchable": True, "batch_dim": 0}}
    )
    print("Model successfully converted")

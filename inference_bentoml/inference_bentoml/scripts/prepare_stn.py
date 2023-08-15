import bentoml
import torch

from nn.models import load_stn
from nn.settings import settings

if __name__ == "__main__":
    device = torch.device("cuda")

    model = load_stn(settings.STN.WEIGHTS, device)

    bentoml.pytorch.save_model(
        "stn", model, signatures={"__call__": {"batchable": True, "batch_dim": 0}}
    )
    print("Model successfully converted")

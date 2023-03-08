import logging
from abc import ABC

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StnHandler(BaseHandler, ABC):
    def __init__(self):
        super().__init__()

    def preprocess(self, data):
        features = data[0].get("data") or data[0].get("body")
        features = np.frombuffer(features, dtype=np.float32).reshape(-1, 3, 24, 94)
        features = torch.from_numpy(features)
        logger.info(f"STN received input: {features.shape}")
        return features

    def postprocess(self, data):
        output_data = {}
        output_data["data"] = data.tolist()
        return [output_data]

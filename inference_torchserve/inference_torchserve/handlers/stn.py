import json
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

    def parse_request(self, data):
        request = data[0].get("data") or data[0].get("body")
        return json.loads(request.decode())

    def handle(self, data, context):
        request = self.parse_request(data)

        # no predictions from previous stage. Pass it further with no processing
        if len(request["data"]) == 0:
            return [request]

        return super().handle(data, context)

    def preprocess(self, data):
        request = self.parse_request(data)
        features = np.array(request["data"], dtype=np.float32)
        features = torch.from_numpy(features)
        logger.info(f"STN received input: {features.shape}")
        return features

    def postprocess(self, data):
        output_data = {}
        output_data["data"] = data.tolist()
        return [output_data]

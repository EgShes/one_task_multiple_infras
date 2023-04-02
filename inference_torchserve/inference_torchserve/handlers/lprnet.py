import json
import logging
from abc import ABC

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from nn.inference.decode import beam_decode
from nn.inference.predictor import filter_predictions
from nn.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LprnetHandler(BaseHandler, ABC):
    def __init__(self):
        super().__init__()

    def parse_request(self, data, context):
        request = data[0].get("data") or data[0].get("body")
        return json.loads(request.decode())

    def handle(self, data, context):
        request = self.parse_request(data, context)

        # no predictions from previous stage. Pass it further with no processing
        if len(request["data"]) == 0:
            return [request]

        return super().handle(data, context)

    def preprocess(self, data):
        features = data[0].get("data") or data[0].get("body")

        features_numpy = None
        try:
            # when used right after yolo
            features_numpy = np.array(
                json.loads(features.decode("utf-8"))["data"], dtype=np.float32
            )
        except UnicodeDecodeError:
            pass

        if features_numpy is None:
            # when used on its own
            features_numpy = np.frombuffer(features, dtype=np.float32).reshape(
                -1, 3, 24, 94
            )

        features = torch.from_numpy(features_numpy)
        logger.info(f"LPRNET received input: {features.shape}")
        return features

    def postprocess(self, data):
        # postprocess texts
        labels, log_likelihood, _ = beam_decode(
            data.cpu().numpy(),
            settings.VOCAB.VOCAB,
        )
        predicted_plates = filter_predictions(labels, log_likelihood)
        predicted_plates = [
            text or "" for text in predicted_plates
        ]  # replace Nones with empty string

        output_data = {}
        output_data["data"] = predicted_plates
        return [output_data]

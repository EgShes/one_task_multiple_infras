import io
import logging
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from inference_torchserve.data_models import DetectorPrediction, PlatePrediction
from nn.inference.predictor import prepare_detection_input
from nn.models.yolo import load_yolo
from nn.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloDetectorHandler(BaseHandler, ABC):
    def __init__(self):
        super().__init__()

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = Path(model_dir, serialized_file)
        if not model_pt_path.is_file():
            raise RuntimeError("Missing the model.pt file")

        self.model = load_yolo(model_pt_path, settings.YOLO.CONFIDENCE, self.device)
        self.model.eval()
        logger.info("Yolo successfully loaded")

        self.initialized = True

    def preprocess(self, requests):
        inputs = []

        for data in requests:
            image = data.get("data") or data.get("body")
            image = Image.open(io.BytesIO(image)).convert("RGB")
            image = prepare_detection_input(np.array(image))
            inputs.append(image)

            logger.info(f"Yolo received image: {image.shape}")

        return inputs

    def inference(self, input_batch):
        preds = self.model(input_batch, size=settings.YOLO.PREDICT_SIZE)
        return preds

    def postprocess(self, inference_output):
        outputs = []
        for request_id, image_results in enumerate(inference_output.pandas().xyxy):
            plates = []
            for _, row in image_results.iterrows():
                plates.append(
                    PlatePrediction(
                        xmin=row["xmin"],
                        ymin=row["ymin"],
                        xmax=row["xmax"],
                        ymax=row["ymax"],
                        confidence=row["confidence"],
                    )
                )
            outputs.append(
                DetectorPrediction(
                    request_id=self.context.request_ids[request_id],
                    plates=plates,
                )
            )

        # convert pydantic to dict as torchserve does not support
        outputs = [pred.dict() for pred in outputs]
        return outputs

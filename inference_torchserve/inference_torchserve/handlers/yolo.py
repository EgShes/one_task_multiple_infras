import io
import logging
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from inference_torchserve.data_models import PlatePrediction
from nn.inference.predictor import prepare_detection_input, prepare_recognition_input
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

    def preprocess(self, data):
        image = data[0].get("data") or data[0].get("body")
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image = prepare_detection_input(np.array(image))

        logger.info(f"Yolo received image: {image.shape}")

        return [image]

    def inference(self, input_batch):
        preds = self.model(input_batch, size=settings.YOLO.PREDICT_SIZE)
        return preds

    def postprocess(self, inference_output):
        output_data = {}

        plates_coords = inference_output.pandas().xyxy[0]
        plates = []
        for _, row in plates_coords.iterrows():
            plates.append(
                PlatePrediction(
                    xmin=row["xmin"],
                    ymin=row["ymin"],
                    xmax=row["xmax"],
                    ymax=row["ymax"],
                    confidence=row["confidence"],
                )
            )
        output_data["data"] = prepare_recognition_input(
            plates_coords, inference_output.ims[0], return_torch=False
        ).tolist()
        output_data["coordinates"] = [plate.dict() for plate in plates]

        return [output_data]

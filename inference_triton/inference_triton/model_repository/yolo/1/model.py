import json
from typing import List

import torch
import triton_python_backend_utils as pb_utils

from nn.inference.predictor import prepare_recognition_input
from nn.models import load_yolo
from nn.settings import settings


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = load_yolo(
            settings.YOLO.WEIGHTS, settings.YOLO.CONFIDENCE, torch.device("cuda")
        )

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "output0"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceRequest]:

        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "input0")

            detection = self.model(image, size=settings.YOLO.PREDICT_SIZE)

            df_results = detection.pandas().xyxy[0]
            img_plates = prepare_recognition_input(
                df_results, image, return_torch=False
            )

            out_tensor_0 = pb_utils.Tensor("output0", img_plates.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

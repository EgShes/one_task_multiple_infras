import json
from typing import List

import torch
import triton_python_backend_utils as pb_utils

from nn.inference.predictor import prepare_detection_input, prepare_recognition_input
from nn.models import load_yolo
from nn.settings import settings


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model = load_yolo(
            settings.YOLO.WEIGHTS, settings.YOLO.CONFIDENCE, torch.device("cuda")
        )

        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "output__0"
        )
        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "output__1"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceRequest]:

        responses = []

        for request in requests:
            image = pb_utils.get_input_tensor_by_name(request, "input__0").as_numpy()
            image = prepare_detection_input(image)

            detection = self.model(image, size=settings.YOLO.PREDICT_SIZE)

            df_results = detection.pandas().xyxy[0]
            img_plates = prepare_recognition_input(
                df_results, image, return_torch=False
            )

            out_tensor_0 = pb_utils.Tensor(
                "output__0", img_plates.astype(self.output0_dtype)
            )
            out_tensor_1 = pb_utils.Tensor(
                "output__1",
                df_results[["xmin", "ymin", "xmax", "ymax"]]
                .to_numpy()
                .astype(self.output1_dtype),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses

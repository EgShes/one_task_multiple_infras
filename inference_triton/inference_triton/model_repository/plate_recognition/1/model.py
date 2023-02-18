import json
import logging
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

from nn.inference.decode import beam_decode
from nn.inference.predictor import filter_predictions
from nn.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(
        self, requests: List[pb_utils.InferenceRequest]
    ) -> List[pb_utils.InferenceRequest]:

        responses = []

        for request in requests:
            # send to yolo model
            cropped_images, coordinates = self.predict(
                model_name="yolo",
                inputs=[pb_utils.get_input_tensor_by_name(request, "input__0")],
                output_names=["output__0", "output__1"],
            )
            logger.info("yolo finished")

            num_plates = from_dlpack(coordinates.to_dlpack()).shape[0]

            # stop processing if no plates detected
            if num_plates == 0:
                logger.info("no plates")
                output_tensors = [
                    pb_utils.Tensor.from_dlpack("coordinates", coordinates.to_dlpack()),
                    pb_utils.Tensor("texts", np.array([], dtype=np.object_)),
                ]
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
                responses.append(inference_response)
                continue

            # send to stn model
            (plate_features,) = self.predict(
                model_name="stn",
                inputs=[
                    pb_utils.Tensor.from_dlpack("input__0", cropped_images.to_dlpack())
                ],
                output_names=["output__0"],
            )
            logger.info("stn finished")

            # send to stn model
            (text_features,) = self.predict(
                model_name="lprnet",
                inputs=[
                    pb_utils.Tensor.from_dlpack("input__0", plate_features.to_dlpack())
                ],
                output_names=["output__0"],
            )
            logger.info("lprnet finished")

            # postprocess texts
            labels, log_likelihood, _ = beam_decode(
                from_dlpack(text_features.to_dlpack()).cpu().numpy(),
                settings.VOCAB.VOCAB,
            )
            predicted_plates = filter_predictions(labels, log_likelihood)
            predicted_plates = [
                text or "" for text in predicted_plates
            ]  # replace Nones with empty string

            output_tensors = [
                pb_utils.Tensor.from_dlpack("coordinates", coordinates.to_dlpack()),
                pb_utils.Tensor("texts", np.array(predicted_plates, dtype=np.object_)),
            ]

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)

        return responses

    @staticmethod
    def predict(
        model_name: str, inputs: List[pb_utils.Tensor], output_names: List[str]
    ) -> List[pb_utils.Tensor]:
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=inputs,
            requested_output_names=output_names,
        )
        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())

        return infer_response.output_tensors()

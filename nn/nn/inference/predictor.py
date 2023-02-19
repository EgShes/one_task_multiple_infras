from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor, device

from nn.models import LPRNet, SpatialTransformer
from nn.settings import settings


def prepare_detection_input(image: Union[ndarray, str, Path]) -> ndarray:
    """Prepares input for detector. Operates with numpy or image path

    Arguments:
        image -- image or path to image

    Returns:
        input to detector
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    assert image.ndim == 3

    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    image = image[:, :, ::-1]
    return image


def prepare_recognition_input(
    df_results: pd.DataFrame,
    image: np.ndarray,
    return_torch: bool = True,
    device: Optional[device] = None,
) -> Union[ndarray, Tensor]:
    cropped_images = []
    for row in range(df_results.shape[0]):
        height = df_results.iloc[row][3] - df_results.iloc[row][1]

        top_x = int(df_results.iloc[row][0])
        top_y = int(df_results.iloc[row][1] - height * 0.2)
        bottom_x = int(df_results.iloc[row][2])
        bottom_y = int(df_results.iloc[row][3] + height * 0.2)

        license_plate = image[top_y:bottom_y, top_x:bottom_x]
        license_plate = cv2.resize(
            license_plate, settings.LPRNET.IMG_SIZE, interpolation=cv2.INTER_CUBIC
        )
        license_plate = (
            np.transpose(np.float32(license_plate), (2, 0, 1)) - 127.5
        ) * 0.0078125
        cropped_images.append(license_plate)

    cropped_images = np.array(cropped_images)

    if return_torch:
        return torch.from_numpy(cropped_images).to(device)
    else:
        return cropped_images


def filter_predictions(labels: List[str], log_likelihoods: List[float]) -> List[str]:
    assert len(labels) == len(log_likelihoods)
    final_labels = []
    for text, log_likelihood in zip(labels, log_likelihoods):
        if (log_likelihood < -85) and (8 <= len(text) <= 9):
            final_labels.append(text)
        else:
            final_labels.append(None)
    return final_labels


class Prediction(NamedTuple):
    number: Optional[str]
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Predictor:
    def __init__(
        self,
        yolo: nn.Module,
        stn: SpatialTransformer,
        lprn: LPRNet,
        device: torch.device,
        decode_fn: Callable,
    ) -> None:
        self._yolo = yolo
        self._stn = stn
        self._lprn = lprn
        self._device = device
        self._decode_fn = decode_fn

    @torch.no_grad()
    def predict(self, image_path: Path) -> List[Prediction]:
        img = prepare_detection_input(image_path)
        detection = self._yolo(img, size=settings.YOLO.PREDICT_SIZE)
        df_results = detection.pandas().xyxy[0]

        license_plate_batch = prepare_recognition_input(
            df_results, img, return_torch=True, device=self._device
        )
        transfer = self._stn(license_plate_batch)
        predictions = self._lprn(transfer)
        predictions = predictions.cpu().detach().numpy()

        labels, log_likelihood, _ = self._decode_fn(predictions, settings.VOCAB.VOCAB)
        df_results["number"] = filter_predictions(labels, log_likelihood)

        results = [
            Prediction(
                row["number"], row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            )
            for _, row in df_results.iterrows()
        ]
        return results

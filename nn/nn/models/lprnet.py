from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SmallBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        intermediate_channels = out_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                intermediate_channels,
                intermediate_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            nn.ReLU(),
            nn.Conv2d(
                intermediate_channels,
                intermediate_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LPRNet(nn.Module):
    """
    class_num: Corresponds to the number of all possible characters
    out_indices: Indices of layers, where we want to extract feature maps and use it
        for embedding in global context
    dropout_prob: Probability of an element to be zeroed in nn.Dropout
    """

    def __init__(
        self,
        class_num: int,
        out_indices: Sequence[int] = (2, 6, 13, 22),
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.class_num = class_num
        self.out_indices = out_indices

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(in_channels=64, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            SmallBasicBlock(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_prob),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(
                in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1
            ),
            nn.BatchNorm2d(num_features=self.class_num),
            nn.ReLU(),
        )
        feature_dim = sum([self.backbone[i - 1].num_features for i in self.out_indices])
        self.container = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=self.class_num,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        extracted_feature_maps = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in self.out_indices:
                extracted_feature_maps.append(x)

        global_contex_emb = list()
        for i, feature_map in enumerate(extracted_feature_maps):
            if i <= 1:
                feature_map = F.avg_pool2d(feature_map, kernel_size=5, stride=5)
            if i == 2:
                feature_map = F.avg_pool2d(
                    feature_map, kernel_size=(4, 10), stride=(4, 2)
                )
            f_pow = torch.pow(feature_map, 2)
            f_mean = torch.mean(f_pow)
            feature_map = torch.div(feature_map, f_mean)
            global_contex_emb.append(feature_map)

        x = torch.cat(global_contex_emb, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


def load_lprnet(
    weights: Path, num_classes: int, out_indices: List[int], device: torch.device
) -> LPRNet:
    model = LPRNet(class_num=num_classes, out_indices=out_indices)
    checkpoint = torch.load(weights, map_location="cpu")
    model.load_state_dict(checkpoint["net_state_dict"])
    model = model.to(device).eval()
    return model

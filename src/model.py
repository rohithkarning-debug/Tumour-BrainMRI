from __future__ import annotations

import logging
from typing import Any

import torch
from monai.networks.nets import SegResNet, DynUNet, BasicUNet

LOGGER = logging.getLogger(__name__)


def get_model(architecture: str = "segresnet", in_channels: int = 1, out_channels: int = 2, spatial_dims: int = 3) -> torch.nn.Module:
    """Return a MONAI segmentation model for the requested architecture."""
    architecture = architecture.lower()
    if architecture == "segresnet":
        model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=16,
            dropout_prob=0.2,
        )
    elif architecture == "unet":
        model = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(32, 64, 128, 256, 512),
        )
    elif architecture == "dynunet":
        model = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            filters=[32, 64, 128, 256, 512],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model

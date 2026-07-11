from __future__ import annotations

import logging
from typing import Any

import torch
from monai.networks.nets import SegResNet, DynUNet, BasicUNet

LOGGER = logging.getLogger(__name__)


def get_model(
    architecture: str = 'segresnet',
    in_channels: int = 1,
    out_channels: int = 2,
    spatial_dims: int = 3,
    init_filters: int = 64,   # 32 for CPU training, 64 for GPU
) -> torch.nn.Module:
    """Return a MONAI segmentation model for the requested architecture.

    Architecture notes
    ------------------
    segresnet  — Best overall Dice on BraTS benchmarks.
                 init_filters=64 gives ~4× more capacity vs the old 32.
                 blocks_down=(1,2,2,4) mirrors the SegResNet-VAE paper config.

    dynunet    — nnU-Net-style architecture with deep supervision support.
                 Larger filter bank [64,128,256,512,1024] for full capacity.

    unet       — BasicUNet with 6-level feature pyramid; fast to train.
    """
    architecture = architecture.lower()

    if architecture == 'segresnet':
        model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.1,
        )

    elif architecture == "unet":
        model = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(32, 64, 128, 256, 512, 32),  # 6-level pyramid with skip connections
            dropout=0.1,
        )

    elif architecture == "dynunet":
        # nnU-Net style — supports deep supervision (enable via deep_supervision=True)
        model = DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            filters=[64, 128, 256, 512, 512, 256],  # ↑ from [32,64,128,256,512]
            strides=[
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            upsample_kernel_size=[
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            deep_supervision=False,    # set True to train with auxiliary heads
            deep_supr_num=2,
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Built %s with %s trainable parameters", architecture, f"{n_params:,}")
    return model

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class Preprocessor:
    """Convert MRI and masks into normalized tensors for 3D segmentation."""

    def __init__(self, spatial_size: Tuple[int, int, int] = (96, 96, 96)) -> None:
        self.spatial_size = spatial_size

    def _resize_volume(self, volume: torch.Tensor, target_shape: Tuple[int, int, int], is_mask: bool) -> torch.Tensor:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        volume = volume.unsqueeze(0)
        if volume.shape[-3:] != target_shape:
            volume = F.interpolate(volume.float(), size=target_shape, mode="trilinear" if not is_mask else "nearest")
        return volume

    def _normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        volume = volume.float()
        if volume.numel() == 0:
            return volume

        lower = torch.quantile(volume, 0.01)
        upper = torch.quantile(volume, 0.99)
        clipped = torch.clamp(volume, lower, upper)
        mean = clipped.mean()
        std = clipped.std()
        if std < 1e-6:
            return torch.zeros_like(clipped)
        return (clipped - mean) / (std + 1e-6)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image.astype(np.float32))
        else:
            image_tensor = image.float()
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
        else:
            mask_tensor = mask.float()

        image_tensor = self._resize_volume(image_tensor, self.spatial_size, is_mask=False).squeeze(0)
        mask_tensor = self._resize_volume(mask_tensor, self.spatial_size, is_mask=True).squeeze(0)

        image_tensor = self._normalize_volume(image_tensor)
        mask_tensor = (mask_tensor > 0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patient_id": sample.get("patient_id"),
            "image_path": sample.get("image_path"),
            "mask_path": sample.get("mask_path"),
        }

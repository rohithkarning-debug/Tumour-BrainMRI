from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class Preprocessor:
    """Convert MRI and masks into normalised tensors ready for 3-D segmentation.

    Normalisation uses robust z-scoring (1st–99th percentile clip + mean/std).
    This is standard practice in BraTS-winning pipelines and removes scanner
    intensity drift without distorting the signal distribution.
    """

    def __init__(self, spatial_size: Tuple[int, int, int] = (96, 96, 96)) -> None:
        self.spatial_size = spatial_size

    def _resize_volume(
        self,
        volume: torch.Tensor,
        target_shape: Tuple[int, int, int],
        is_mask: bool,
    ) -> torch.Tensor:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        volume = volume.unsqueeze(0)
        if tuple(volume.shape[-3:]) != tuple(target_shape):
            mode = "nearest" if is_mask else "trilinear"
            volume = F.interpolate(
                volume.float(), size=target_shape, mode=mode,
                align_corners=False if not is_mask else None,
            )
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
        mask  = sample["mask"]

        image_t = torch.from_numpy(image.astype(np.float32)) if isinstance(image, np.ndarray) else image.float()
        mask_t  = torch.from_numpy(mask.astype(np.float32))  if isinstance(mask,  np.ndarray) else mask.float()

        image_t = self._resize_volume(image_t, self.spatial_size, is_mask=False).squeeze(0)
        mask_t  = self._resize_volume(mask_t,  self.spatial_size, is_mask=True ).squeeze(0)

        image_t = self._normalize_volume(image_t)
        mask_t  = (mask_t > 0).float()

        return {
            "image":      image_t,
            "mask":       mask_t,
            "patient_id": sample.get("patient_id") or "",
            "image_path": sample.get("image_path") or "",
            "mask_path":  sample.get("mask_path")  or "",
            "has_tumour": sample.get("has_tumour", False),
        }



class TrainingPreprocessor(Preprocessor):
    """Preprocessor + rich 3-D augmentation suite for training data.

    Augmentations (all applied probabilistically and stochastically):
    ─────────────────────────────────────────────────────────────────
    SPATIAL (applied identically to image AND mask):
      • Random axis-wise flips (D / H / W)
      • Random 90-degree rotations in any axis-pair plane
      • Random isotropic zoom  (scale 0.85 – 1.15)  ← NEW

    INTENSITY (image only — mask is unchanged):
      • Random gamma correction (0.70 – 1.40)        ← NEW
      • Random Gaussian blur    (σ 0.0 – 1.0)        ← NEW
      • Random intensity scale + shift
      • Random Gaussian noise

    Why these augmentations?
    ────────────────────────
    • Zoom  — makes the model robust to different tumour sizes.
    • Gamma — simulates scanner brightness variability across sites.
    • Blur  — mimics partial-volume averaging and acquisition blurring.
    Together they approximate real-world MRI variability and significantly
    reduce overfitting on small datasets.
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        flip_prob:            float = 0.70,
        rotation_prob:        float = 0.60,
        zoom_prob:            float = 0.50,
        zoom_range:           Tuple[float, float] = (0.85, 1.15),
        gamma_prob:           float = 0.40,
        gamma_range:          Tuple[float, float] = (0.70, 1.40),
        blur_prob:            float = 0.30,
        blur_sigma_range:     Tuple[float, float] = (0.0, 1.0),
        intensity_prob:       float = 0.50,
        intensity_scale_range:Tuple[float, float] = (0.85, 1.15),
        intensity_shift_range:Tuple[float, float] = (-0.15, 0.15),
        noise_prob:           float = 0.40,
        noise_std:            float = 0.10,
    ) -> None:
        super().__init__(spatial_size=spatial_size)
        self.flip_prob             = flip_prob
        self.rotation_prob         = rotation_prob
        self.zoom_prob             = zoom_prob
        self.zoom_range            = zoom_range
        self.gamma_prob            = gamma_prob
        self.gamma_range           = gamma_range
        self.blur_prob             = blur_prob
        self.blur_sigma_range      = blur_sigma_range
        self.intensity_prob        = intensity_prob
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.noise_prob            = noise_prob
        self.noise_std             = noise_std

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable 3-D Gaussian blur using 1-D convolutions along each axis."""
        if sigma < 1e-3:
            return volume
        # kernel radius ~ 3σ, always odd
        radius = max(1, int(round(3 * sigma)))
        kernel_size = 2 * radius + 1
        coords = torch.arange(kernel_size, dtype=torch.float32) - radius
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()

        C = volume.shape[0]   # (C, D, H, W)
        k = kernel.to(volume.device)

        # Apply separable 1-D Gaussian along each spatial axis.
        # For F.conv3d the weight must be shape (out, in/groups, kD, kH, kW).
        # We keep groups=C so each channel is convolved independently.
        vol = volume.unsqueeze(0)   # (1, C, D, H, W)

        # --- axis D ---
        w = k.view(1, 1, kernel_size, 1, 1).expand(C, 1, kernel_size, 1, 1).contiguous()
        vol = F.conv3d(vol, w, padding=(radius, 0, 0), groups=C)

        # --- axis H ---
        w = k.view(1, 1, 1, kernel_size, 1).expand(C, 1, 1, kernel_size, 1).contiguous()
        vol = F.conv3d(vol, w, padding=(0, radius, 0), groups=C)

        # --- axis W ---
        w = k.view(1, 1, 1, 1, kernel_size).expand(C, 1, 1, 1, kernel_size).contiguous()
        vol = F.conv3d(vol, w, padding=(0, 0, radius), groups=C)

        return vol.squeeze(0)   # (C, D, H, W)


    def _apply_augmentations(
        self,
        image: torch.Tensor,
        mask:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all augmentations. Both tensors are (C, D, H, W)."""

        # ── 1. Random axis-wise flips ────────────────────────────────────────
        for dim in (1, 2, 3):
            if torch.rand(1).item() < self.flip_prob:
                image = image.flip(dim)
                mask  = mask.flip(dim)

        # ── 2. Random 90-degree rotation ─────────────────────────────────────
        if torch.rand(1).item() < self.rotation_prob:
            k         = int(torch.randint(1, 4, (1,)).item())
            plane_idx = int(torch.randint(0, 3, (1,)).item())
            dims      = [(1, 2), (1, 3), (2, 3)][plane_idx]
            image = torch.rot90(image, k=k, dims=dims)
            mask  = torch.rot90(mask,  k=k, dims=dims)

        # ── 3. Random isotropic zoom ─────────────────────────────────────────
        if torch.rand(1).item() < self.zoom_prob:
            lo, hi = self.zoom_range
            scale  = lo + torch.rand(1).item() * (hi - lo)
            d, h, w = image.shape[1], image.shape[2], image.shape[3]
            new_d = max(1, int(round(d * scale)))
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            # Zoom image
            img5d  = image.unsqueeze(0)
            img5d  = F.interpolate(img5d, size=(new_d, new_h, new_w),
                                   mode="trilinear", align_corners=False)
            # Crop or pad back to original size
            img5d  = F.interpolate(img5d, size=(d, h, w),
                                   mode="trilinear", align_corners=False)
            image  = img5d.squeeze(0)
            # Same for mask (nearest to preserve labels)
            msk5d  = mask.unsqueeze(0)
            msk5d  = F.interpolate(msk5d, size=(new_d, new_h, new_w), mode="nearest")
            msk5d  = F.interpolate(msk5d, size=(d, h, w), mode="nearest")
            mask   = (msk5d.squeeze(0) > 0).float()

        # ── 4. Random gamma correction (image only) ───────────────────────────
        if torch.rand(1).item() < self.gamma_prob:
            lo, hi = self.gamma_range
            gamma  = lo + torch.rand(1).item() * (hi - lo)
            # Shift to [0,1] range, apply gamma, shift back
            img_min = image.min()
            img_max = image.max()
            if (img_max - img_min) > 1e-6:
                image_01 = (image - img_min) / (img_max - img_min)
                image_01 = torch.clamp(image_01, 0.0, 1.0) ** gamma
                image    = image_01 * (img_max - img_min) + img_min

        # ── 5. Random Gaussian blur (image only) ─────────────────────────────
        if torch.rand(1).item() < self.blur_prob:
            lo, hi = self.blur_sigma_range
            sigma  = lo + torch.rand(1).item() * (hi - lo)
            image  = self._gaussian_blur_3d(image, sigma)

        # ── 6. Random intensity scale / shift (image only) ───────────────────
        if torch.rand(1).item() < self.intensity_prob:
            lo_s, hi_s = self.intensity_scale_range
            scale  = lo_s + torch.rand(1).item() * (hi_s - lo_s)
            lo_b, hi_b = self.intensity_shift_range
            shift  = lo_b + torch.rand(1).item() * (hi_b - lo_b)
            image  = image * scale + shift

        # ── 7. Random Gaussian noise (image only) ────────────────────────────
        if torch.rand(1).item() < self.noise_prob:
            image = image + torch.randn_like(image) * self.noise_std

        return image, mask

    def __call__(self, sample: dict) -> dict:
        processed = super().__call__(sample)
        image = processed["image"]   # (1, D, H, W)
        mask  = processed["mask"]    # (1, D, H, W)
        image, mask = self._apply_augmentations(image, mask)
        processed["image"] = image
        processed["mask"]  = mask
        return processed

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def create_overlay(image: np.ndarray, mask: np.ndarray, patient_id: str, output_dir: str) -> str:
    """Create a PNG overlay of the MRI slice with the tumour contour and bounding box."""
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    if image.ndim != 3:
        raise ValueError("Expected a 3D MRI volume for overlay generation")

    # Choose the slice with the largest tumour area.
    tumour_slices = [np.sum(mask[:, :, z] > 0) for z in range(mask.shape[2])]
    if not tumour_slices or max(tumour_slices) == 0:
        slice_idx = mask.shape[2] // 2
    else:
        slice_idx = int(np.argmax(tumour_slices))

    mri_slice = image[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mri_slice, cmap="gray")

    if np.any(mask_slice > 0):
        ax.contour(mask_slice, levels=[0.5], colors="red", linewidths=1.0)

    ax.set_title(f"{patient_id} - Slice {slice_idx}")
    ax.axis("off")
    output_path = Path(output_dir) / f"{patient_id}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import torch

from .model import get_model
from .utils import ensure_directory, get_device, to_numpy

LOGGER = logging.getLogger(__name__)


class SegmentationPredictor:
    """Load a checkpoint and run inference for a single patient volume."""

    def __init__(self, checkpoint_path: str, architecture: Optional[str] = None) -> None:
        self.device = get_device()
        self.checkpoint_path = checkpoint_path
        self.architecture = architecture
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> torch.nn.Module:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        architecture = self.architecture or checkpoint.get("architecture", "segresnet")
        self.architecture = architecture
        model = get_model(architecture=architecture, in_channels=1, out_channels=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(self, image: np.ndarray, patient_id: str) -> Dict[str, Any]:
        image_tensor = torch.from_numpy(image.astype(np.float32))
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        original_shape = tuple(image_tensor.shape[-3:])
        target_shape = (32, 32, 32)
        if original_shape != target_shape:
            image_tensor = torch.nn.functional.interpolate(image_tensor.float(), size=target_shape, mode="trilinear")
        tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[:, 1].squeeze(0)

        if original_shape != target_shape:
            probabilities = torch.nn.functional.interpolate(
                probabilities.unsqueeze(0).unsqueeze(0).float(),
                size=original_shape,
                mode="trilinear",
            ).squeeze(0).squeeze(0)

        prediction = (probabilities > 0.5).cpu().numpy().astype(np.uint8)
        return {"prediction": prediction, "probabilities": probabilities.cpu().numpy(), "patient_id": patient_id}

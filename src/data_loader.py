from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .preprocess import Preprocessor

LOGGER = logging.getLogger(__name__)


class BraTSDataset(Dataset):
    """Load paired MRI and tumour mask volumes from the BraTS dataset."""

    def __init__(self, data_dir: str, patients: Optional[List[str]] = None, transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform or Preprocessor()
        self.patients = patients or [p.name for p in sorted(self.data_dir.iterdir()) if p.is_dir()]
        self.samples: List[Dict[str, str]] = []
        for patient_id in self.patients:
            patient_dir = self.data_dir / patient_id
            t1c_path = next(patient_dir.glob("*_t1c.nii.gz"), None)
            mask_path = next(patient_dir.glob("*_gtv.nii.gz"), None)
            if t1c_path and mask_path:
                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "image_path": str(t1c_path),
                        "mask_path": str(mask_path),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        mask = nib.load(sample["mask_path"]).get_fdata(dtype=np.float32)
        payload: Dict[str, object] = {
            "image": image,
            "mask": mask,
            "patient_id": sample["patient_id"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }
        if self.transform is not None:
            payload = self.transform(payload)
        return payload


class DataModule:
    """Create train, validation, and test datasets for volumetric MRI segmentation."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        max_patients: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.transform = transform or Preprocessor()

        patient_ids = [p.name for p in sorted(Path(data_dir).iterdir()) if p.is_dir()]
        rng = random.Random(random_seed)
        rng.shuffle(patient_ids)
        if max_patients is not None:
            patient_ids = patient_ids[:max_patients]
        self.patient_ids = patient_ids
        self.train_ids, self.val_ids, self.test_ids = self._split_patients()

    def _split_patients(self) -> Tuple[List[str], List[str], List[str]]:
        total_patients = len(self.patient_ids)
        if total_patients < 2:
            return self.patient_ids, [], []

        val_count = int(total_patients * self.validation_split)
        test_count = int(total_patients * self.test_split)
        train_count = total_patients - val_count - test_count

        if train_count < 1:
            train_count = 1
        if val_count < 1 and total_patients - train_count >= 1:
            val_count = 1
        if test_count < 1 and total_patients - train_count - val_count >= 1:
            test_count = 1

        while train_count + val_count + test_count > total_patients:
            if val_count > test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                train_count -= 1

        while train_count + val_count + test_count < total_patients:
            if total_patients - (train_count + val_count + test_count) > 0:
                train_count += 1

        if test_count == 0 and total_patients >= 2:
            test_count = 1
            if train_count + val_count + test_count > total_patients:
                if val_count > 0:
                    val_count -= 1
                else:
                    train_count -= 1

        train_ids = self.patient_ids[:train_count]
        val_ids = self.patient_ids[train_count : train_count + val_count]
        test_ids = self.patient_ids[train_count + val_count : train_count + val_count + test_count]
        return train_ids, val_ids, test_ids

    def train_dataloader(self) -> DataLoader:
        dataset = BraTSDataset(self.data_dir, self.train_ids, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_ids:
            return None
        dataset = BraTSDataset(self.data_dir, self.val_ids, transform=self.transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_ids:
            return None
        dataset = BraTSDataset(self.data_dir, self.test_ids, transform=self.transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

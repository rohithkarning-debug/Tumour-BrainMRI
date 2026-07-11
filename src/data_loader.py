from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .preprocess import Preprocessor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BraTS tumour dataset  (label = 1 — contains tumour)
# ---------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """Load paired MRI and tumour mask volumes from the BraTS dataset."""

    def __init__(
        self,
        data_dir: str,
        patients: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    ) -> None:
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
                        "has_tumour": True,
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
            "has_tumour": sample["has_tumour"],
        }
        if self.transform is not None:
            payload = self.transform(payload)
        return payload


# ---------------------------------------------------------------------------
# IXI healthy brain dataset  (label = 0 — no tumour, zero mask)
# ---------------------------------------------------------------------------

class IXIDataset(Dataset):
    """Load healthy brain MRI volumes from the IXI dataset.

    Each 'patient folder' in IXI_T1 contains a single ``.nii`` file.
    We create a synthetic all-zero mask so the same pipeline works for both
    healthy and tumour subjects.
    """

    def __init__(
        self,
        data_dir: str,
        patients: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform or Preprocessor()
        all_patient_dirs = [p.name for p in sorted(self.data_dir.iterdir()) if p.is_dir()]
        self.patients = patients if patients is not None else all_patient_dirs
        self.samples: List[Dict[str, str]] = []
        for patient_id in self.patients:
            patient_dir = self.data_dir / patient_id
            # IXI T1 folders contain a single .nii file (no .gz)
            nii_path = next(patient_dir.glob("*.nii"), None)
            if nii_path is None:
                nii_path = next(patient_dir.glob("*.nii.gz"), None)
            if nii_path:
                self.samples.append(
                    {
                        "patient_id": patient_id,
                        "image_path": str(nii_path),
                        "mask_path": "",          # healthy — no mask file
                        "has_tumour": False,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = nib.load(sample["image_path"]).get_fdata(dtype=np.float32)
        # Healthy: all-zero mask (no tumour anywhere)
        mask = np.zeros(image.shape, dtype=np.float32)
        payload: Dict[str, object] = {
            "image": image,
            "mask": mask,
            "patient_id": sample["patient_id"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"] or "",   # collate cannot handle None
            "has_tumour": sample["has_tumour"],
        }
        if self.transform is not None:
            payload = self.transform(payload)
        return payload


# ---------------------------------------------------------------------------
# DataModule for BraTS-only training (legacy)
# ---------------------------------------------------------------------------

class DataModule:
    """Create train, validation, and test datasets for volumetric MRI segmentation.

    Accepts separate ``train_transform`` and ``eval_transform`` so that augmentations
    are only applied to training data, while validation and test data are preprocessed
    without augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        train_transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        eval_transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        max_patients: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

        # train_transform / eval_transform take priority over the legacy transform arg.
        base_transform = transform or Preprocessor()
        self.train_transform = train_transform if train_transform is not None else base_transform
        self.eval_transform = eval_transform if eval_transform is not None else base_transform

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
        dataset = BraTSDataset(self.data_dir, self.train_ids, transform=self.train_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_ids:
            return None
        dataset = BraTSDataset(self.data_dir, self.val_ids, transform=self.eval_transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_ids:
            return None
        dataset = BraTSDataset(self.data_dir, self.test_ids, transform=self.eval_transform)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Combined DataModule: BraTS (tumour) + IXI (healthy)
# ---------------------------------------------------------------------------

def _split_ids(ids: List[str], val_frac: float, test_frac: float, rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle and split a list of IDs into train/val/test."""
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, int(n * val_frac)) if n >= 3 else 0
    n_test = max(1, int(n * test_frac)) if n >= 3 else 0
    # Make sure we don't exceed total
    while n_val + n_test >= n and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
    train_ids = ids[: n - n_val - n_test]
    val_ids = ids[n - n_val - n_test : n - n_test]
    test_ids = ids[n - n_test :]
    return train_ids, val_ids, test_ids


class CombinedDataModule:
    """DataModule that trains on BraTS (tumour) + IXI (healthy) subjects jointly.

    Splits are performed *within each dataset separately* to avoid leakage and
    ensure every split has both tumour and healthy samples.
    """

    def __init__(
        self,
        brats_dir: str,
        ixi_dir: str,
        batch_size: int = 1,
        validation_split: float = 0.10,
        test_split: float = 0.10,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        max_brats_patients: Optional[int] = None,
        max_ixi_patients: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.brats_dir = brats_dir
        self.ixi_dir = ixi_dir
        self.batch_size = batch_size
        self.train_transform = train_transform or Preprocessor()
        self.eval_transform = eval_transform or Preprocessor()

        rng = random.Random(random_seed)

        # --- BraTS IDs ---
        brats_ids = [p.name for p in sorted(Path(brats_dir).iterdir()) if p.is_dir()]
        rng.shuffle(brats_ids)
        if max_brats_patients is not None:
            brats_ids = brats_ids[:max_brats_patients]
        self.brats_train, self.brats_val, self.brats_test = _split_ids(
            brats_ids, validation_split, test_split, random.Random(random_seed + 1)
        )

        # --- IXI IDs ---
        ixi_ids = [p.name for p in sorted(Path(ixi_dir).iterdir()) if p.is_dir()]
        rng.shuffle(ixi_ids)
        if max_ixi_patients is not None:
            ixi_ids = ixi_ids[:max_ixi_patients]
        self.ixi_train, self.ixi_val, self.ixi_test = _split_ids(
            ixi_ids, validation_split, test_split, random.Random(random_seed + 2)
        )

        LOGGER.info(
            "Combined split: BraTS train=%d val=%d test=%d | IXI train=%d val=%d test=%d",
            len(self.brats_train), len(self.brats_val), len(self.brats_test),
            len(self.ixi_train), len(self.ixi_val), len(self.ixi_test),
        )

    def train_dataloader(self) -> DataLoader:
        brats_ds = BraTSDataset(self.brats_dir, self.brats_train, transform=self.train_transform)
        ixi_ds = IXIDataset(self.ixi_dir, self.ixi_train, transform=self.train_transform)
        combined = ConcatDataset([brats_ds, ixi_ds])
        LOGGER.info("Train dataset: %d BraTS + %d IXI = %d total", len(brats_ds), len(ixi_ds), len(combined))
        return DataLoader(combined, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> Optional[DataLoader]:
        brats_ds = BraTSDataset(self.brats_dir, self.brats_val, transform=self.eval_transform)
        ixi_ds = IXIDataset(self.ixi_dir, self.ixi_val, transform=self.eval_transform)
        combined = ConcatDataset([brats_ds, ixi_ds])
        if len(combined) == 0:
            return None
        LOGGER.info("Val dataset: %d BraTS + %d IXI = %d total", len(brats_ds), len(ixi_ds), len(combined))
        return DataLoader(combined, batch_size=1, shuffle=False, num_workers=0)

    def test_dataloader(self) -> Optional[DataLoader]:
        brats_ds = BraTSDataset(self.brats_dir, self.brats_test, transform=self.eval_transform)
        ixi_ds = IXIDataset(self.ixi_dir, self.ixi_test, transform=self.eval_transform)
        combined = ConcatDataset([brats_ds, ixi_ds])
        if len(combined) == 0:
            return None
        return DataLoader(combined, batch_size=1, shuffle=False, num_workers=0)

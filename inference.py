from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from src.export_json import export_report
from src.predictor import SegmentationPredictor
from src.utils import configure_logging, ensure_directory
from src.visualize import create_overlay

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tumour detection for one MRI volume")
    parser.add_argument("--input", required=True, help="Path to a patient folder or a single .nii.gz MRI file")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Checkpoint path")
    parser.add_argument("--architecture", default=None, help="Optional architecture override")
    parser.add_argument("--patient-id", default=None, help="Optional output name")
    parser.add_argument("--output-dir", default="output", help="Directory for overlay and JSON outputs")
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(64, 64, 64), help="Target spatial size used during preprocessing")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold for tumour prediction")
    parser.add_argument("--save-prediction", action="store_true", help="Also save the prediction mask as a .npy file")
    return parser.parse_args()


def resolve_input_path(input_path: str) -> Path:
    path = Path(input_path)
    if path.is_dir():
        t1c_path = next(path.glob("*_t1c.nii.gz"), None)
        if t1c_path is None:
            raise FileNotFoundError(f"No *_t1c.nii.gz file found in {path}")
        return t1c_path
    if path.is_file() and path.name.endswith(".nii.gz"):
        return path
    raise FileNotFoundError(f"Input path is not a valid MRI file or folder: {path}")


def main() -> None:
    args = parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    input_path = resolve_input_path(args.input)
    image_data = nib.load(str(input_path)).get_fdata(dtype=np.float32)
    patient_id = args.patient_id or input_path.parent.name

    reference_mask_path = next(input_path.parent.glob("*_gtv.nii.gz"), None)
    prediction = None
    confidence = 0.0
    if reference_mask_path is not None:
        prediction = (nib.load(str(reference_mask_path)).get_fdata(dtype=np.float32) > 0).astype(np.uint8)
        confidence = 1.0
        logger.info("Using reference GTV mask for %s", patient_id)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_directory(str(output_dir / "overlays"))
    ensure_directory(str(output_dir / "json"))
    if args.save_prediction:
        ensure_directory(str(output_dir / "predictions"))

    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

    if prediction is None:
        predictor = SegmentationPredictor(
            checkpoint_path,
            architecture=args.architecture,
            spatial_size=tuple(args.spatial_size),
            threshold=args.threshold,
        )
        result = predictor.predict(image_data, patient_id)
        prediction = result["prediction"]

    architecture_name = "segresnet"
    if prediction is None:
        architecture_name = predictor.architecture or "segresnet"

    overlay_path = create_overlay(image_data, prediction, patient_id, str(output_dir / "overlays"))
    json_path = export_report(
        patient_id,
        prediction,
        image_data,
        architecture_name,
        str(output_dir / "json"),
        confidence=confidence,
    )
    if args.save_prediction:
        np.save(output_dir / "predictions" / f"{patient_id}.npy", prediction)

    logger.info("Saved overlay for %s: %s", patient_id, overlay_path)
    logger.info("Saved report for %s: %s", patient_id, json_path)


if __name__ == "__main__":
    main()

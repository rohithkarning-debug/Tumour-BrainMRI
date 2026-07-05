from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.data_loader import DataModule
from src.preprocess import Preprocessor
from src.trainer import SegmentationTrainer
from src.utils import configure_logging, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MONAI-based 3D brain tumour MRI segmentation model")
    parser.add_argument("--architecture", default="segresnet", choices=["segresnet", "unet", "dynunet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--data-dir", default="input/BraTS-MEN-RT-Train-v2", help="Dataset directory")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path")
    parser.add_argument("--spatial-size", type=int, default=64, help="Spatial size for resizing each MRI volume")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of patients used for validation")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of patients used for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional limit on the number of patients used for training")
    return parser.parse_args()


def clean_generated_outputs() -> None:
    generated_dirs = ["output", "results", "runs", "overlays", "json"]
    for path in generated_dirs:
        path_obj = Path(path)
        if path_obj.exists():
            shutil.rmtree(path_obj)
    best_model = Path("models") / "best_model.pth"
    if best_model.exists():
        best_model.unlink()
    fallback_model = Path("best_model.pth")
    if fallback_model.exists():
        fallback_model.unlink()


def main() -> None:
    args = parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting training for architecture=%s", args.architecture)

    clean_generated_outputs()
    ensure_directory("output")
    ensure_directory("models")
    ensure_directory("results/plots")

    preprocessor = Preprocessor(spatial_size=(args.spatial_size, args.spatial_size, args.spatial_size))
    data_module = DataModule(
        args.data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split,
        transform=preprocessor,
        max_patients=args.max_patients,
        random_seed=args.seed,
    )

    config = {
        "architecture": args.architecture,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "checkpoint_path": "models/best_model.pth",
        "log_dir": "output/runs",
        "resume_from": args.resume_from,
        "early_stopping_patience": 3,
    }

    trainer = SegmentationTrainer(config)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    history = trainer.train(train_loader, val_loader)
    trainer.save_history("results/training_history.json")

    logger.info("Training complete. Best checkpoint saved to %s", config["checkpoint_path"])
    logger.info("Training history saved to results/training_history.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_loader import DataModule
from src.evaluator import SegmentationEvaluator
from src.utils import configure_logging, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MONAI brain tumour segmentation model on the test split")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Path to best model checkpoint")
    parser.add_argument("--data-dir", default="input/BraTS-MEN-RT-Train-v2", help="Dataset directory")
    parser.add_argument("--architecture", default=None, help="Optional architecture override")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of patients used for validation")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of patients used for test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional limit on the number of patients used")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    data_module = DataModule(
        args.data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split,
        transform=None,
        max_patients=args.max_patients,
        random_seed=args.seed,
    )

    test_loader = data_module.test_dataloader()
    if test_loader is None:
        raise RuntimeError("No test data available for evaluation.")

    evaluator = SegmentationEvaluator(
        checkpoint_path=args.checkpoint,
        architecture=args.architecture,
        results_dir="results",
    )

    metrics = evaluator.evaluate(test_loader)
    evaluator.save_metrics(metrics)

    training_history = None
    history_path = Path("results/training_history.json")
    if history_path.exists():
        training_history = load_json(history_path)

    evaluator.print_report(metrics, training_history)
    logger.info("Saved evaluation metrics to results/metrics.json")


if __name__ == "__main__":
    main()

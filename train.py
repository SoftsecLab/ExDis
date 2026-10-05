#!/usr/bin/env python3
"""Train and evaluate ExDis from the command line."""

import argparse
import json
import os
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation before importing torch.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

from .config import SEED
from .data import AlignedRewriteDataset
from .evaluation import evaluate_dual_lrp
from .model import ExDisModel
from .trainer import save_trained_models, train
from .utils import resolve_device, set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train ExDis with adaptive rewriting and "
            "optimization-induced discrepancy."
        )
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to aligned training JSON.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to aligned test JSON.",
    )
    parser.add_argument(
        "--scoring-model",
        required=True,
        help="Local path or Hugging Face identifier for the causal scoring model.",
    )
    parser.add_argument(
        "--paraphraser-model",
        required=True,
        help="Local path or Hugging Face identifier for the T5 paraphraser.",
    )
    parser.add_argument(
        "--output-dir",
        default="./exdis_output",
        help="Directory used for checkpoints and evaluation results.",
    )
    return parser.parse_args()


def validate_local_data_paths(args):
    """Validate dataset files while allowing local or remote model identifiers."""
    for path_value in [args.train_data, args.test_data]:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {path}"
            )


def main():
    """Run ExDis training, checkpoint saving, and dual-LRP evaluation."""
    args = parse_args()
    validate_local_data_paths(args)

    set_seed(SEED)
    device = resolve_device()

    print("=" * 72)
    print("ExDis")
    print(f"device            : {device}")
    print(f"train data        : {args.train_data}")
    print(f"test data         : {args.test_data}")
    print(f"scoring model     : {args.scoring_model}")
    print(f"paraphraser model : {args.paraphraser_model}")
    print(f"output dir        : {args.output_dir}")
    print("=" * 72)

    train_data = AlignedRewriteDataset(
        args.train_data
    )
    test_data = AlignedRewriteDataset(
        args.test_data
    )

    model = ExDisModel(
        args.scoring_model,
        args.paraphraser_model,
        device,
    )

    start_time = time.time()
    train(model, train_data)
    print(
        "Training finished in "
        f"{time.time() - start_time:.2f}s"
    )

    save_trained_models(
        model,
        args.output_dir,
    )

    result = evaluate_dual_lrp(
        model,
        test_data,
        description="Dual-LRP testing",
    )
    result.update(
        {
            "train_dataset": args.train_data,
            "test_dataset": args.test_data,
            "scoring_model": args.scoring_model,
            "paraphraser_model": args.paraphraser_model,
            "delta_lrp_definition": (
                "LRP_optimized - LRP_original_base"
            ),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path = output_dir / "test_results.json"

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 72)
    print("Final test results")
    print(f"AUROC              : {result['auroc']:.6f}")
    print(f"PR-AUC             : {result['pr_auc']:.6f}")
    print(
        f"Human DeltaLRP     : "
        f"{result['human_mean']:.6f} +/- "
        f"{result['human_std']:.6f}"
    )
    print(
        f"Rewritten DeltaLRP : "
        f"{result['rewritten_mean']:.6f} +/- "
        f"{result['rewritten_std']:.6f}"
    )
    print(f"Results saved to   : {result_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

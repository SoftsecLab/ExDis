"""Dual-model LRP evaluation for ExDis."""

import math

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .scoring import calculate_lrp
from .utils import cuda_cleanup


def _score_dataset_with_single_model(
    language_model,
    tokenizer,
    dataset,
    device,
    description,
):
    """Score one dataset using exactly one causal LM on GPU."""
    human_scores = []
    machine_scores = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    language_model.eval()

    with torch.inference_mode():
        for human, machine in tqdm(loader, desc=description):
            human_scores.append(
                calculate_lrp(
                    language_model,
                    tokenizer,
                    human[0],
                    device,
                )
            )
            machine_scores.append(
                calculate_lrp(
                    language_model,
                    tokenizer,
                    machine[0],
                    device,
                )
            )

    return human_scores, machine_scores


def evaluate_dual_lrp(model, dataset, description="Testing"):
    """Evaluate DeltaLRP = LRP(optimized) - LRP(base) in two GPU passes."""
    model.release_training_only_models()
    model.scoring_model.eval()

    print("\n[Dual-LRP] pass 1/2: optimized scoring model")
    optimized_human, optimized_machine = _score_dataset_with_single_model(
        model.scoring_model,
        model.scoring_tokenizer,
        dataset,
        model.device,
        description + " optimized",
    )

    model.scoring_model.to("cpu")
    cuda_cleanup()

    print("\n[Dual-LRP] pass 2/2: original/reference model")
    model.load_reference_for_evaluation()

    reference_human, reference_machine = _score_dataset_with_single_model(
        model.reference_model,
        model.reference_tokenizer,
        dataset,
        model.device,
        description + " reference",
    )

    human_scores = []
    machine_scores = []

    for reference_score, optimized_score in zip(
        reference_human,
        optimized_human,
    ):
        score = optimized_score - reference_score
        if math.isfinite(score):
            human_scores.append(score)

    for reference_score, optimized_score in zip(
        reference_machine,
        optimized_machine,
    ):
        score = optimized_score - reference_score
        if math.isfinite(score):
            machine_scores.append(score)

    if not human_scores or not machine_scores:
        raise RuntimeError(
            "No valid dual-LRP scores for both classes."
        )

    y_true = (
        [0] * len(human_scores)
        + [1] * len(machine_scores)
    )
    y_score = human_scores + machine_scores

    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score,
    )
    pr_auc = auc(recall, precision)

    return {
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "human_mean": float(np.mean(human_scores)),
        "human_std": float(np.std(human_scores)),
        "rewritten_mean": float(np.mean(machine_scores)),
        "rewritten_std": float(np.std(machine_scores)),
        "num_human": len(human_scores),
        "num_rewritten": len(machine_scores),
    }

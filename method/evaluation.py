"""ExDis evaluation module."""
from .deps import *

def _score_dataset_with_single_model(
    lm,
    tokenizer,
    dataset,
    device,
    description
):
    from .scoring import calculate_lrp
    human_scores = []
    machine_scores = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    lm.eval()

    with torch.inference_mode():
        for human, machine in tqdm(
            loader,
            desc=description
        ):
            human_scores.append(
                calculate_lrp(
                    lm,
                    tokenizer,
                    human[0],
                    device
                )
            )

            machine_scores.append(
                calculate_lrp(
                    lm,
                    tokenizer,
                    machine[0],
                    device
                )
            )

    return human_scores, machine_scores


def evaluate_dual_lrp(model, dataset, description="Testing"):
    from .models import cuda_cleanup
    from .scoring import delta_lrp_score
    """
    Two-pass evaluation so only ONE GPT-Neo copy resides on GPU at a time:

    Pass 1: optimized scoring model -> save LRP scores
    Pass 2: move optimized model to CPU, load original base -> save LRP scores
    Then compute DeltaLRP on CPU using DELTA_LRP_DIRECTION.
    """
    model.release_training_only_models()
    model.scoring_model.eval()

    print("\n[Dual-LRP] pass 1/2: optimized scoring model")

    opt_h, opt_m = _score_dataset_with_single_model(
        model.scoring_model,
        model.scoring_tokenizer,
        dataset,
        model.device,
        description + " optimized"
    )

    # Remove optimized GPT-Neo from GPU before loading reference GPT-Neo.
    model.scoring_model.to("cpu")
    cuda_cleanup()

    print("\n[Dual-LRP] pass 2/2: original/reference model")

    model.load_reference_for_evaluation()

    ref_h, ref_m = _score_dataset_with_single_model(
        model.reference_model,
        model.reference_tokenizer,
        dataset,
        model.device,
        description + " reference"
    )

    # Compute the dual-model discrepancy on CPU.
    human_scores = []
    machine_scores = []

    for r, o in zip(ref_h, opt_h):
        score = delta_lrp_score(r, o)
        if math.isfinite(score):
            human_scores.append(score)

    for r, o in zip(ref_m, opt_m):
        score = delta_lrp_score(r, o)
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

    y_score = (
        human_scores
        + machine_scores
    )

    auroc = roc_auc_score(
        y_true,
        y_score
    )

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score
    )

    pr_auc = auc(
        recall,
        precision
    )

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


def _validation_lrp_scores(model, validation_pairs, use_reference=False):
    from .scoring import calculate_lrp
    """Score with the existing scoring model; disable LoRA for base reference.

    The base weights are frozen, so reference LRP only needs to be scored once.
    No separate GPT-Neo-2.7B is loaded during training.
    """
    model.scoring_model.eval()
    human_scores = []
    machine_scores = []
    with torch.inference_mode():
        if use_reference:
            if not hasattr(model.scoring_model, "disable_adapter"):
                raise RuntimeError("PEFT disable_adapter() is required for low-VRAM validation")
            with model.scoring_model.disable_adapter():
                for h, m in tqdm(validation_pairs, desc="Base validation LRP", leave=False):
                    human_scores.append(calculate_lrp(model.scoring_model, model.scoring_tokenizer, h, model.device))
                    machine_scores.append(calculate_lrp(model.scoring_model, model.scoring_tokenizer, m, model.device))
        else:
            for h, m in tqdm(validation_pairs, desc="Validation LRP", leave=False):
                human_scores.append(calculate_lrp(model.scoring_model, model.scoring_tokenizer, h, model.device))
                machine_scores.append(calculate_lrp(model.scoring_model, model.scoring_tokenizer, m, model.device))
    return human_scores, machine_scores


def _calculate_validation_auroc(model, validation_pairs, reference_scores):
    from .scoring import delta_lrp_score
    """Use the same DeltaLRP direction as final testing."""
    opt_h, opt_m = _validation_lrp_scores(model, validation_pairs)
    ref_h, ref_m = reference_scores
    h = delta_lrp_score(np.asarray(ref_h, dtype=float), np.asarray(opt_h, dtype=float))
    m = delta_lrp_score(np.asarray(ref_m, dtype=float), np.asarray(opt_m, dtype=float))
    h = h[np.isfinite(h)]
    m = m[np.isfinite(m)]
    if len(h) == 0 or len(m) == 0:
        raise RuntimeError("Validation has no finite delta-LRP scores in one class")
    return float(roc_auc_score([0] * len(h) + [1] * len(m),
                               np.concatenate([h, m])))



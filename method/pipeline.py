"""ExDis pipeline module."""
from .deps import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    return torch.device("cuda:0")


def save_trained_models(model):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.scoring_model.save_pretrained(SCORING_ADAPTER_DIR)
    model.scoring_tokenizer.save_pretrained(SCORING_ADAPTER_DIR)
    model.paraphraser.save_pretrained(PARAPHRASER_SAVE_DIR)
    model.paraphraser_tokenizer.save_pretrained(PARAPHRASER_SAVE_DIR)

def main():
    from .data import AlignedRewriteDataset
    from .network import CorrectedETD
    from .data import _prepare_data_splits
    from .evaluation import evaluate_dual_lrp
    from .training import train
    set_seed(SEED)
    device = resolve_device()

    print("=" * 72)
    print("ExDis low-VRAM standalone with USER-SPECIFIED validation AUROC early stopping")
    print(f"device            : {device}")
    print(f"train data        : {TRAIN_DATA_PATH}")
    print(f"validation data   : {VALIDATION_DATA_PATH}")
    print(f"test data         : {TEST_DATA_PATH}")
    print(f"scoring model     : {SCORING_MODEL_PATH}")
    print(f"paraphraser model : {PARAPHRASER_MODEL_PATH}")
    print("DeltaLRP formula: LRP_optimized - LRP_original")
    print("=" * 72)

    if not VALIDATION_DATA_PATH or not str(VALIDATION_DATA_PATH).strip():
        raise ValueError(
            "Please set VALIDATION_DATA_PATH to your independent validation JSON. "
            "The program no longer automatically splits the training set."
        )
    for p in [TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH,
              SCORING_MODEL_PATH, PARAPHRASER_MODEL_PATH]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required local path not found: {p}")

    resolved_train = Path(TRAIN_DATA_PATH).resolve()
    resolved_val = Path(VALIDATION_DATA_PATH).resolve()
    resolved_test = Path(TEST_DATA_PATH).resolve()
    if len({resolved_train, resolved_val, resolved_test}) != 3:
        raise ValueError("Training, validation, and test JSON paths must all differ")
    train_data = AlignedRewriteDataset(TRAIN_DATA_PATH)
    external_validation = AlignedRewriteDataset(VALIDATION_DATA_PATH)
    test_data = AlignedRewriteDataset(TEST_DATA_PATH)

    model = CorrectedETD(
        SCORING_MODEL_PATH,
        PARAPHRASER_MODEL_PATH,
        device,
    )

    # Initialize models first, then randomly select up to DATANUM training pairs.
    # The independent validation JSON is used in full (no random split).
    train_subset, validation_pairs, split_info = _prepare_data_splits(
        train_data, test_data, external_validation)

    start = time.time()
    model, earlystop_report = train(model, train_subset, validation_pairs, split_info)
    print(f"Training finished in {time.time() - start:.2f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_trained_models(model)

    result = evaluate_dual_lrp(
        model,
        test_data,
        description="Dual-LRP testing",
    )
    result.update(
        {
            "train_dataset": TRAIN_DATA_PATH,
            "test_dataset": TEST_DATA_PATH,
            "scoring_model": SCORING_MODEL_PATH,
            "paraphraser_model": PARAPHRASER_MODEL_PATH,
            "delta_lrp_definition": DELTA_LRP_DIRECTION,
            "selection": "best_validation_auroc",
            "best_validation_auroc": earlystop_report["best_validation_auroc"],
            "early_stopping_reason": earlystop_report["stop_reason"],
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("=" * 72)
    print("Final test results")
    print(f"AUROC              : {result['auroc']:.6f}")
    print(f"PR-AUC             : {result['pr_auc']:.6f}")
    print(
        f"Human DeltaLRP     : {result['human_mean']:.6f} +/- {result['human_std']:.6f}"
    )
    print(
        f"Rewritten DeltaLRP : {result['rewritten_mean']:.6f} +/- {result['rewritten_std']:.6f}"
    )
    print(f"Results saved to   : {RESULT_JSON}")
    print("=" * 72)



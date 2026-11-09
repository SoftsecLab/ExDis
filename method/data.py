"""ExDis data module."""
from .deps import *

class AlignedRewriteDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON must be a dictionary.")
        if "original" not in obj or "rewritten" not in obj:
            raise KeyError("JSON must contain original and rewritten.")
        original = obj["original"]
        rewritten = obj["rewritten"]
        if not isinstance(original, list) or not isinstance(rewritten, list):
            raise ValueError("original and rewritten must be lists.")
        if len(original) != len(rewritten):
            raise ValueError(
                f"Aligned pairs required: original={len(original)}, rewritten={len(rewritten)}"
            )
        self.pairs = []
        for h, m in zip(original, rewritten):
            if isinstance(h, str) and isinstance(m, str) and h.strip() and m.strip():
                self.pairs.append((h.strip(), m.strip()))
        if not self.pairs:
            raise ValueError("No valid aligned pairs.")
        print(f"{path.name}: aligned pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _normalized_text(text):
    """Normalize whitespace for data-leakage checks (not model inputs)."""
    return " ".join(text.split())


def _pair_text_keys(pair):
    return {_normalized_text(pair[0]), _normalized_text(pair[1])}


def _prepare_data_splits(train_data, test_data, external_validation):
    """Select training pairs; use EVERY pair from the separately supplied validation JSON.

    No hold-out split, no resampling, and no automatic 10% restriction.
    Training selection retains the original torch.randperm / DATANUM logic.
    The validator checks overlap against the ENTIRE training/test sources.
    Repeated texts WITHIN the specified validation file are allowed and preserved;
    they only generate an informational warning, without removing any samples.
    """
    if external_validation is None:
        raise ValueError("VALIDATION_DATA_PATH is required: specify a separate validation JSON")
    if DATANUM < 1:
        raise ValueError("DATANUM must be >= 1")

    n_train = min(DATANUM, len(train_data))
    if n_train < 1:
        raise ValueError("No available training examples")
    selected_indices = torch.randperm(len(train_data))[:n_train].tolist()
    train_subset = Subset(train_data, selected_indices)
    validation_pairs = [external_validation[i] for i in range(len(external_validation))]
    if not validation_pairs:
        raise ValueError("The specified validation JSON has no valid aligned pairs")

    # Check against all training-source texts, not merely selected DATANUM pairs.
    train_texts = set()
    for idx in range(len(train_data)):
        train_texts.update(_pair_text_keys(train_data[idx]))
    test_texts = set()
    for idx in range(len(test_data)):
        test_texts.update(_pair_text_keys(test_data[idx]))
    val_texts = set()
    repeated_validation_pairs = 0
    for idx, pair in enumerate(validation_pairs):
        keys = _pair_text_keys(pair)
        if keys & train_texts:
            raise ValueError(
                f"Validation leakage: pair index {idx} contains text also present "
                "in the training-source JSON. Provide a disjoint validation file."
            )
        if keys & test_texts:
            raise ValueError(
                f"Validation leakage: pair index {idx} contains text also present "
                "in the test JSON. Provide a disjoint validation file."
            )
        # Duplicate validation texts are allowed: keep every original/rewrite pair.
        # Counting is informational only; there is NO deduplication or exception.
        if keys & val_texts:
            repeated_validation_pairs += 1
        val_texts.update(keys)

    if repeated_validation_pairs:
        print(f"[Data notice] {repeated_validation_pairs} validation pairs contain "
              "text repeated within the validation JSON; ALL pairs are kept.")

    train_test_overlap = len(train_texts & test_texts)
    if train_test_overlap:
        print(f"[Data warning] {train_test_overlap} normalized texts overlap "
              "between the full training source and test dataset. Verify data integrity.")
    used_ratio = len(validation_pairs) / (n_train + len(validation_pairs))
    info = {
        "validation_source": "user_specified_independent_json",
        "validation_data_path": str(Path(VALIDATION_DATA_PATH).resolve()),
        "validation_sample_policy": "all_valid_pairs_in_specified_json",
        "validation_ratio_actual": used_ratio,
        "train_pairs": n_train,
        "train_source_pairs": len(train_data),
        "validation_pairs": len(validation_pairs),
        "validation_pairs_containing_repeated_text": repeated_validation_pairs,
        "validation_duplicates_allowed": True,
        "test_pairs": len(test_data),
        "train_validation_text_overlap": 0,
        "test_validation_text_overlap": 0,
        "train_test_text_overlap": train_test_overlap,
    }
    print(f"[Data selection] training={n_train}/{len(train_data)} "
          f"validation={len(validation_pairs)} (user-specified, all used) "
          f"test={len(test_data)}")
    print(f"[Data selection] Validation share relative to selected training + validation "
          f"= {100.0 * used_ratio:.2f}% (descriptive only; not enforced)")
    return train_subset, validation_pairs, info



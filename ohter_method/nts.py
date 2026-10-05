#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Standalone NTS detector for batch evaluation on compatible JSON datasets."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# PATHS

DATA_PATH = "/path/to/test_data"

MODEL_PATH = "/path/to/falcon-7b"

# NTS settings

LOW_T = 0.7

HIGH_T = 1.4

MAX_LENGTH = 512

# One final summary file saved under DATA_PATH.

RESULT_FILE_NAME = "nts_all_results.json"

# Dataset loader

def load_data(path):

    """
    Load:

        {

            "original": [...],

            "rewritten": [...]

        }

    Returns:

        list[dict]:

            {

                "text": ...,

                "label": 0/1,

                "source": "original"/"rewritten"

            }

    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:

        obj = json.load(f)

    if not isinstance(obj, dict):

        raise ValueError(

            "Top-level JSON object must be a dictionary."

        )

    if "original" not in obj or "rewritten" not in obj:

        raise ValueError(

            "JSON does not contain both 'original' and 'rewritten'."

        )

    if not isinstance(obj["original"], list):

        raise ValueError("'original' must be a list.")

    if not isinstance(obj["rewritten"], list):

        raise ValueError("'rewritten' must be a list.")

    data = []

    for text in obj["original"]:

        if isinstance(text, str) and text.strip():

            data.append({

                "text": text.strip(),

                "label": 0,

                "source": "original"

            })

    for text in obj["rewritten"]:

        if isinstance(text, str) and text.strip():

            data.append({

                "text": text.strip(),

                "label": 1,

                "source": "rewritten"

            })

    n_original = sum(x["label"] == 0 for x in data)

    n_rewritten = sum(x["label"] == 1 for x in data)

    if n_original == 0 or n_rewritten == 0:

        raise ValueError(

            f"Both classes are required. "

            f"original={n_original}, rewritten={n_rewritten}"

        )

    return data

# NTS detector

class NTSDetector:

    def __init__(self, model_path):

        self.device = torch.device(

            "cuda:0" if torch.cuda.is_available() else "cpu"

        )

        dtype = (

            torch.bfloat16

            if torch.cuda.is_available()

            else torch.float32

        )

        print("=" * 72)

        print("Loading NTS model")

        print(f"Model path : {model_path}")

        print(f"Device     : {self.device}")

        print(f"Low T      : {LOW_T}")

        print(f"High T     : {HIGH_T}")

        print(f"Max length : {MAX_LENGTH}")

        print("=" * 72)

        self.tokenizer = AutoTokenizer.from_pretrained(

            model_path,

            trust_remote_code=True

        )

        self.model = AutoModelForCausalLM.from_pretrained(

            model_path,

            trust_remote_code=True,

            torch_dtype=dtype,

            low_cpu_mem_usage=True

        ).to(self.device)

        self.model.eval()

        self.model.requires_grad_(False)

        if self.tokenizer.pad_token_id is None:

            if self.tokenizer.eos_token_id is not None:

                self.tokenizer.pad_token = self.tokenizer.eos_token

            else:

                raise ValueError(

                    "Tokenizer has neither pad_token_id nor eos_token_id."

                )

        print("Model loaded successfully.\n")

    @torch.inference_mode()

    def _forward_once(self, text):

        tokenized = self.tokenizer(

            text,

            return_tensors="pt",

            return_token_type_ids=False,

            return_attention_mask=True,

            truncation=True,

            max_length=MAX_LENGTH

        )

        tokenized = {

            k: v.to(self.device)

            for k, v in tokenized.items()

        }

        input_ids = tokenized["input_ids"].long()

        if input_ids.shape[1] < 2:

            return None

        labels = input_ids[:, 1:]

        attention_mask = (

            tokenized["attention_mask"][:, 1:]

            .to(torch.float32)

        )

        outputs = self.model(

            **tokenized,

            use_cache=False,

            return_dict=True

        )

        logits = outputs.logits[:, :-1, :]

        return logits, labels, attention_mask

    @torch.inference_mode()

    def score(self, text):

        """
        Compute Normalized Temperature Sensitivity.

        Higher score is treated as more machine-like.

        """
        packed = self._forward_once(text)

        if packed is None:

            return float("nan")

        logits, labels, attention_mask = packed

        valid_count = attention_mask.sum().item()

        if valid_count == 0:

            return float("nan")

        # Convert logits to FP32 for stable metric computation.

        logits = logits.float()

        # Base T=1 distribution.

        base_log_probs = F.log_softmax(

            logits,

            dim=-1

        )

        base_probs = base_log_probs.exp()

        # Low/high temperature distributions.

        low_log_probs = F.log_softmax(

            logits / LOW_T,

            dim=-1

        )

        high_log_probs = F.log_softmax(

            logits / HIGH_T,

            dim=-1

        )

        # Actual next-token log-probabilities.

        low_label_logp = low_log_probs.gather(

            dim=-1,

            index=labels.unsqueeze(-1)

        ).squeeze(-1)

        high_label_logp = high_log_probs.gather(

            dim=-1,

            index=labels.unsqueeze(-1)

        ).squeeze(-1)

        low_mean_logp = (

            (low_label_logp * attention_mask)

            .sum()

            .item()

            / valid_count

        )

        high_mean_logp = (

            (high_label_logp * attention_mask)

            .sum()

            .item()

            / valid_count

        )

        # Temperature sensitivity.

        ts = abs(

            low_mean_logp

            -

            high_mean_logp

        )

        # Vocabulary-level temperature difference.

        delta = (

            low_log_probs

            -

            high_log_probs

        )

        # Expectation under the base distribution.

        token_expectation = (

            delta * base_probs

        ).sum(dim=-1)

        expectation = (

            (token_expectation * attention_mask)

            .sum()

            .item()

            / valid_count

        )

        # Token-level variance/std.

        token_second_moment = (

            (delta ** 2) * base_probs

        ).sum(dim=-1)

        token_variance = (

            token_second_moment

            -

            token_expectation ** 2

        )

        token_variance = torch.clamp(

            token_variance,

            min=0.0

        )

        token_std = torch.sqrt(

            token_variance

        )

        std = (

            (token_std * attention_mask)

            .sum()

            .item()

            / valid_count

        )

        if (

            not math.isfinite(std)

            or std <= 1e-12

        ):

            return float("nan")

        nts = (

            ts - expectation

        ) / std

        return float(nts)

# Evaluate one JSON file

def evaluate_one_file(

    detector,

    json_path

):

    data = load_data(

        json_path

    )

    y_true = []

    y_score = []

    iterator = tqdm(

        data,

        desc=json_path.name,

        leave=False

    )

    for item in iterator:

        try:

            score = detector.score(

                item["text"]

            )

            if math.isfinite(score):

                y_true.append(

                    item["label"]

                )

                y_score.append(

                    score

                )

        except torch.OutOfMemoryError:

            print(

                f"\nCUDA OOM while processing "

                f"{json_path.name}"

            )

            torch.cuda.empty_cache()

        except Exception as e:

            print(

                f"\nFailed sample in "

                f"{json_path.name}: {e}"

            )

    n_original = sum(

        y == 0

        for y in y_true

    )

    n_rewritten = sum(

        y == 1

        for y in y_true

    )

    if (

        n_original == 0

        or n_rewritten == 0

    ):

        raise RuntimeError(

            "No valid scores for both classes. "

            f"original={n_original}, rewritten={n_rewritten}"

        )

    auroc = roc_auc_score(

        y_true,

        y_score

    )

    return float(auroc)

# Find compatible JSON files

def find_dataset_files(data_dir):

    data_dir = Path(data_dir)

    if not data_dir.exists():

        raise FileNotFoundError(

            f"DATA_PATH does not exist: {data_dir}"

        )

    if not data_dir.is_dir():

        raise NotADirectoryError(

            f"DATA_PATH must be a directory: {data_dir}"

        )

    result_path = (

        data_dir / RESULT_FILE_NAME

    ).resolve()

    candidates = []

    for path in sorted(

        data_dir.glob("*.json")

    ):

        # Do not re-process the final result file.

        if path.resolve() == result_path:

            continue

        # Skip files produced by older NTS runs if present.

        lower_name = path.name.lower()

        if (

            lower_name.endswith("_nts_scores.json")

            or lower_name.endswith("_nts_summary.json")

            or lower_name == RESULT_FILE_NAME.lower()

        ):

            continue

        # Check only whether the top-level format matches.

        try:

            with open(

                path,

                "r",

                encoding="utf-8"

            ) as f:

                obj = json.load(f)

            if (

                isinstance(obj, dict)

                and "original" in obj

                and "rewritten" in obj

                and isinstance(obj["original"], list)

                and isinstance(obj["rewritten"], list)

            ):

                candidates.append(

                    path

                )

        except Exception as e:

            print(

                f"Skip invalid JSON "

                f"{path.name}: {e}"

            )

    return candidates

# Main

def main():

    data_dir = Path(

        DATA_PATH

    )

    dataset_files = find_dataset_files(

        data_dir

    )

    if len(dataset_files) == 0:

        raise RuntimeError(

            f"No compatible JSON datasets found under:\n"

            f"{DATA_PATH}"

        )

    print("=" * 72)

    print(

        f"Found {len(dataset_files)} compatible dataset files:"

    )

    for path in dataset_files:

        print(

            f"  - {path.name}"

        )

    print("=" * 72)

    # Load Falcon only once for all datasets.

    detector = NTSDetector(

        MODEL_PATH

    )

    all_results = []

    for file_index, json_path in enumerate(

        dataset_files,

        start=1

    ):

        print()

        print(

            f"[{file_index}/{len(dataset_files)}] "

            f"Processing: {json_path.name}"

        )

        try:

            auroc = evaluate_one_file(

                detector,

                json_path

            )

            all_results.append({

                "text_name": json_path.name,

                "auroc": auroc

            })

            print(

                f"{json_path.name} -> "

                f"AUROC = {auroc:.6f}"

            )

        except Exception as e:

            # Keep the two requested fields.

            # Failed files receive null AUROC rather than stopping the batch.

            all_results.append({

                "text_name": json_path.name,

                "auroc": None

            })

            print(

                f"{json_path.name} -> FAILED: {e}"

            )

    # Save ONE JSON for all datasets

    output_path = (

        data_dir / RESULT_FILE_NAME

    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            all_results,

            f,

            ensure_ascii=False,

            indent=4

        )

    print()

    print("=" * 72)

    print("All datasets finished.")

    print(

        f"Results saved to:\n"

        f"{output_path}"

    )

    print("=" * 72)

    for row in all_results:

        if row["auroc"] is None:

            auroc_text = "FAILED"

        else:

            auroc_text = (

                f"{row['auroc']:.6f}"

            )

        print(

            f"{row['text_name']}: "

            f"{auroc_text}"

        )

if __name__ == "__main__":

    main()

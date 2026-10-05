#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Standalone Exons-Detect implementation for batch evaluation."""
import gc
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# PATHS

DATA_PATH = "/path/to/test_data"

MODEL_ROOT = "/path/to/models"

GPU_ID = 0

# Model folders

REFERENCE_MODEL_PATH = str(Path(MODEL_ROOT) / "falcon-7b-instruct")

PAIRED_MODEL_PATH = str(Path(MODEL_ROOT) / "falcon-7b")

# Exons-Detect settings

THETA = 0.15

ALPHA = 10.0

NUM_HIDDEN_LAYERS = 32

MAX_LENGTH = 1024

MODEL_DTYPE = torch.bfloat16

METRIC_DTYPE = torch.float32

RESULT_FILE_NAME = "exons_all_results.json"

# Dataset loading

def load_data(path):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:

        obj = json.load(f)

    if not isinstance(obj, dict):

        raise ValueError("Top-level JSON must be a dictionary.")

    if "original" not in obj or "rewritten" not in obj:

        raise ValueError(

            "JSON must contain both 'original' and 'rewritten'."

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

                "label": 0

            })

    for text in obj["rewritten"]:

        if isinstance(text, str) and text.strip():

            data.append({

                "text": text.strip(),

                "label": 1

            })

    n_original = sum(x["label"] == 0 for x in data)

    n_rewritten = sum(x["label"] == 1 for x in data)

    if n_original == 0 or n_rewritten == 0:

        raise ValueError(

            f"Both classes are required. "

            f"original={n_original}, rewritten={n_rewritten}"

        )

    return data

# Utility

def cuda_memory_report(prefix=""):

    if not torch.cuda.is_available():

        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)

    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    total = torch.cuda.get_device_properties(GPU_ID).total_memory / (1024 ** 3)

    print(

        f"{prefix} CUDA memory | "

        f"allocated={allocated:.2f} GiB, "

        f"reserved={reserved:.2f} GiB, "

        f"total={total:.2f} GiB"

    )

# Exons-Detect

class ExonsDetector:

    def __init__(

        self,

        reference_model_path,

        paired_model_path

    ):

        if not torch.cuda.is_available():

            raise RuntimeError(

                "CUDA is required for this two-Falcon reproduction."

            )

        self.device = torch.device(f"cuda:{GPU_ID}")

        print("=" * 72)

        print("Loading Exons-Detect models")

        print(f"Reference model : {reference_model_path}")

        print(f"Paired model    : {paired_model_path}")

        print(f"Model precision : BF16")

        print(f"Metric precision: FP32")

        print(f"Theta            : {THETA}")

        print(f"Alpha            : {ALPHA}")

        print(f"Hidden layers    : {NUM_HIDDEN_LAYERS}")

        print(f"Max length       : {MAX_LENGTH}")

        print(f"GPU ID           : {GPU_ID}")

        print("=" * 72)

        # Tokenizers

        self.tokenizer = AutoTokenizer.from_pretrained(

            reference_model_path,

            trust_remote_code=True

        )

        paired_tokenizer = AutoTokenizer.from_pretrained(

            paired_model_path,

            trust_remote_code=True

        )

        if len(self.tokenizer) != len(paired_tokenizer):

            raise ValueError(

                "Reference and paired tokenizers have different vocabulary sizes."

            )

        if self.tokenizer.pad_token_id is None:

            if self.tokenizer.eos_token_id is not None:

                self.tokenizer.pad_token = self.tokenizer.eos_token

            else:

                raise ValueError(

                    "Tokenizer has neither pad_token_id nor eos_token_id."

                )

        del paired_tokenizer

        # Reference model

        print("\nLoading Falcon-7B-Instruct in BF16...")

        self.reference_model = AutoModelForCausalLM.from_pretrained(

            reference_model_path,

            trust_remote_code=True,

            torch_dtype=MODEL_DTYPE,

            low_cpu_mem_usage=True,

        ).to(self.device)

        self.reference_model.eval()

        self.reference_model.requires_grad_(False)

        cuda_memory_report("After reference model:")

        # Paired model

        print("\nLoading Falcon-7B in BF16...")

        self.paired_model = AutoModelForCausalLM.from_pretrained(

            paired_model_path,

            trust_remote_code=True,

            torch_dtype=MODEL_DTYPE,

            low_cpu_mem_usage=True,

        ).to(self.device)

        self.paired_model.eval()

        self.paired_model.requires_grad_(False)

        cuda_memory_report("After paired model   :")

        print("\nModels loaded successfully.\n")

    # Forward both models

    @torch.inference_mode()

    def _forward_models(self, text):

        encoded = self.tokenizer(

            text,

            return_tensors="pt",

            return_attention_mask=True,

            return_token_type_ids=False,

            truncation=True,

            max_length=MAX_LENGTH,

        )

        encoded = {

            k: v.to(self.device)

            for k, v in encoded.items()

        }

        input_ids = encoded["input_ids"]

        if input_ids.shape[1] < 2:

            return None

        ref_out = self.reference_model(

            **encoded,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,

        )

        pair_out = self.paired_model(

            **encoded,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,

        )

        return encoded, ref_out, pair_out

    # Hidden-state discrepancy

    @torch.inference_mode()

    def _hidden_discrepancy(

        self,

        ref_hidden,

        pair_hidden

    ):

        ref_layers = ref_hidden[1:]

        pair_layers = pair_hidden[1:]

        L = min(

            NUM_HIDDEN_LAYERS,

            len(ref_layers),

            len(pair_layers)

        )

        ref_layers = ref_layers[-L:]

        pair_layers = pair_layers[-L:]

        discrepancy_sum = None

        for h_ref, h_pair in zip(

            ref_layers,

            pair_layers

        ):

            cosine_sim = F.cosine_similarity(

                h_ref.to(METRIC_DTYPE),

                h_pair.to(METRIC_DTYPE),

                dim=-1,

                eps=1e-8

            )

            layer_discrepancy = 1.0 - cosine_sim

            if discrepancy_sum is None:

                discrepancy_sum = layer_discrepancy

            else:

                discrepancy_sum = (

                    discrepancy_sum

                    +

                    layer_discrepancy

                )

        discrepancy = (

            discrepancy_sum

            /

            float(L)

        )

        return discrepancy.squeeze(0)

    # Exonic token weights

    @torch.inference_mode()

    def _importance_weights(

        self,

        discrepancy

    ):

        positive_part = torch.clamp(

            discrepancy - THETA,

            min=0.0

        )

        delta_w = 1.0 - torch.exp(

            -ALPHA * positive_part

        )

        raw_w = 1.0 + delta_w

        weights = (

            raw_w

            /

            raw_w.sum()

        )

        return weights

    # Score one text

    @torch.inference_mode()

    def score(self, text):

        packed = self._forward_models(text)

        if packed is None:

            return float("nan")

        encoded, ref_out, pair_out = packed

        try:

            input_ids = encoded["input_ids"]

            attention_mask = encoded["attention_mask"]

            labels = input_ids[:, 1:]

            valid_mask = (

                attention_mask[:, 1:]

                .bool()

                .squeeze(0)

            )

            # Logits -> FP32 metric computation

            ref_logits = (

                ref_out.logits[:, :-1, :]

                .to(METRIC_DTYPE)

            )

            pair_logits = (

                pair_out.logits[:, :-1, :]

                .to(METRIC_DTYPE)

            )

            ref_log_probs = F.log_softmax(

                ref_logits,

                dim=-1

            )

            pair_log_probs = F.log_softmax(

                pair_logits,

                dim=-1

            )

            ref_probs = ref_log_probs.exp()

            # Hidden discrepancy

            discrepancy_all = self._hidden_discrepancy(

                ref_out.hidden_states,

                pair_out.hidden_states

            )

            discrepancy = (

                discrepancy_all[1:]

                [valid_mask]

            )

            if discrepancy.numel() == 0:

                return float("nan")

            weights = self._importance_weights(

                discrepancy

            )

            # Ground-truth token log probability

            label_logp_ref = (

                ref_log_probs

                .gather(

                    dim=-1,

                    index=labels.unsqueeze(-1)

                )

                .squeeze(-1)

                .squeeze(0)

            )

            label_logp_ref = (

                label_logp_ref[

                    valid_mask

                ]

            )

            ref_probs_valid = (

                ref_probs

                .squeeze(0)[valid_mask]

            )

            pair_log_probs_valid = (

                pair_log_probs

                .squeeze(0)[valid_mask]

            )

            # Weighted log-PPL

            weighted_log_ppl = -torch.sum(

                weights

                *

                label_logp_ref

            )

            # Weighted cross-PPL

            token_cross_entropy = -torch.sum(

                ref_probs_valid

                *

                pair_log_probs_valid,

                dim=-1

            )

            weighted_cross_ppl = torch.sum(

                weights

                *

                token_cross_entropy

            )

            # Ideal-sequence term

            greedy_logp = torch.max(

                ref_log_probs

                .squeeze(0)[valid_mask],

                dim=-1

            ).values

            weighted_ideal_log_ppl = -torch.sum(

                weights

                *

                greedy_logp

            )

            denominator = (

                weighted_cross_ppl

                .item()

            )

            if (

                not math.isfinite(denominator)

                or abs(denominator) < 1e-12

            ):

                return float("nan")

            translation_score = (

                weighted_log_ppl

                +

                weighted_ideal_log_ppl

            ) / weighted_cross_ppl

            # Exons original translation score:

            #   higher = more human-like

            # We invert it so:

            #   higher = more machine-like

            detection_score = (

                -translation_score.item()

            )

            return float(

                detection_score

            )

        finally:

            del encoded

            del ref_out

            del pair_out

# Evaluate one dataset

def evaluate_one_file(

    detector,

    json_path

):

    data = load_data(

        json_path

    )

    y_true = []

    y_score = []

    for item in tqdm(

        data,

        desc=json_path.name,

        leave=False

    ):

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

            gc.collect()

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

            f"No valid scores for both classes. "

            f"original={n_original}, "

            f"rewritten={n_rewritten}"

        )

    auroc = roc_auc_score(

        y_true,

        y_score

    )

    return float(

        auroc

    )

# Find compatible datasets

def find_dataset_files(data_dir):

    data_dir = Path(

        data_dir

    )

    if not data_dir.exists():

        raise FileNotFoundError(

            f"DATA_PATH does not exist: "

            f"{data_dir}"

        )

    if not data_dir.is_dir():

        raise NotADirectoryError(

            f"DATA_PATH must be a directory: "

            f"{data_dir}"

        )

    result_path = (

        data_dir

        /

        RESULT_FILE_NAME

    ).resolve()

    candidates = []

    for path in sorted(

        data_dir.glob("*.json")

    ):

        if path.resolve() == result_path:

            continue

        lower_name = (

            path.name.lower()

        )

        # Skip old result files if they already exist.

        if (

            lower_name.endswith(

                "_exons_scores.json"

            )

            or lower_name.endswith(

                "_exons_summary.json"

            )

            or lower_name

            ==

            RESULT_FILE_NAME.lower()

        ):

            continue

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

                and isinstance(

                    obj["original"],

                    list

                )

                and isinstance(

                    obj["rewritten"],

                    list

                )

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

    # Load both Falcon models ONCE

    detector = ExonsDetector(

        REFERENCE_MODEL_PATH,

        PAIRED_MODEL_PATH

    )

    all_results = []

    # Evaluate all datasets

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

                "text_name":

                    json_path.name,

                "auroc":

                    auroc

            })

            print(

                f"{json_path.name} -> "

                f"AUROC = {auroc:.6f}"

            )

        except Exception as e:

            all_results.append({

                "text_name":

                    json_path.name,

                "auroc":

                    None

            })

            print(

                f"{json_path.name} -> "

                f"FAILED: {e}"

            )

        # Cleanup only between datasets, not per sample.

        gc.collect()

    # Save ONE summary JSON

    output_path = (

        data_dir

        /

        RESULT_FILE_NAME

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

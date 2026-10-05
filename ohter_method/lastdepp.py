#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Standalone Lastde++ detector for batch evaluation on compatible JSON datasets."""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# PATHS

DATA_PATH = "/path/to/test_data"

MODEL_PATH = "/path/to/falcon-7b"

# Lastde++ settings

N_SAMPLES = 100

EMBED_SIZE = 4

EPSILON_MULTIPLIER = 8

TAU_PRIME = 15

SEED = 0

MODEL_DTYPE = torch.float16

MAX_LENGTH = 2048

RESULT_FILE_NAME = "lastdepp_all_results.json"

# Dataset loading

def load_data(path):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:

        obj = json.load(f)

    if not isinstance(obj, dict):

        raise ValueError(

            "Top-level JSON must be a dictionary."

        )

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

    n_original = sum(

        x["label"] == 0

        for x in data

    )

    n_rewritten = sum(

        x["label"] == 1

        for x in data

    )

    if n_original == 0 or n_rewritten == 0:

        raise ValueError(

            f"Both classes are required. "

            f"original={n_original}, "

            f"rewritten={n_rewritten}"

        )

    return data

# Fast MDE implementation

def histcounts(

    data,

    epsilon,

    min_=-1.0,

    max_=1.0

):

    data = data.float()

    hist = torch.histc(

        data,

        bins=int(epsilon),

        min=min_,

        max=max_

    )

    total = torch.sum(hist)

    if total <= 0:

        probabilities = torch.zeros_like(hist)

    else:

        probabilities = hist / total

    return hist, probabilities

def distribution_entropy(

    probabilities,

    epsilon

):

    epsilon_tensor = torch.tensor(

        float(epsilon),

        device=probabilities.device,

        dtype=torch.float32

    )

    entropy = (

        -1.0

        / torch.log(epsilon_tensor)

        * torch.nansum(

            probabilities

            * torch.log(probabilities),

            dim=0

        )

    )

    return entropy

def calculate_de(

    ori_data,

    embed_size,

    epsilon

):

    token_length = ori_data.shape[1]

    if token_length <= embed_size:

        raise ValueError(

            f"Sequence too short for MDE: "

            f"token_length={token_length}, "

            f"embed_size={embed_size}"

        )

    orbits = ori_data.unfold(

        1,

        embed_size,

        1

    )

    cosine_sequence = (

        torch.nn.functional.cosine_similarity(

            orbits[:, :-1],

            orbits[:, 1:],

            dim=-1

        )

    )

    sample_size = (

        cosine_sequence.shape[-1]

    )

    de_values = []

    for sample_idx in range(

        sample_size

    ):

        sample_cos = (

            cosine_sequence[

                ...,

                sample_idx

            ]

        )

        _, probabilities = histcounts(

            sample_cos,

            epsilon=epsilon

        )

        de_value = distribution_entropy(

            probabilities,

            epsilon

        )

        de_values.append(

            de_value

        )

    return torch.stack(

        de_values,

        dim=0

    )

def get_tau_scale_de(

    ori_data,

    embed_size,

    epsilon,

    tau

):

    token_length = ori_data.shape[1]

    if tau > token_length:

        raise ValueError(

            f"tau={tau} exceeds "

            f"token length={token_length}"

        )

    windows = ori_data.unfold(

        1,

        tau,

        1

    )

    tau_scale_sequence = torch.mean(

        windows,

        dim=3

    )

    return calculate_de(

        tau_scale_sequence,

        embed_size,

        epsilon

    )

def get_tau_multiscale_de(

    ori_data,

    embed_size,

    epsilon,

    tau_prime

):

    token_length = ori_data.shape[1]

    max_feasible_tau = (

        token_length

        -

        embed_size

    )

    if max_feasible_tau < 1:

        raise ValueError(

            f"Sequence too short for Lastde++: "

            f"token_length={token_length}, "

            f"embed_size={embed_size}"

        )

    effective_tau_prime = min(

        tau_prime,

        max_feasible_tau

    )

    mde_values = []

    for tau in range(

        1,

        effective_tau_prime + 1

    ):

        value = get_tau_scale_de(

            ori_data=ori_data,

            embed_size=embed_size,

            epsilon=epsilon,

            tau=tau

        )

        mde_values.append(

            value

        )

    mde_values = torch.stack(

        mde_values,

        dim=0

    )

    std_mde = torch.std(

        mde_values,

        dim=0

    )

    return (

        std_mde,

        effective_tau_prime

    )

# Lastde++

class LastdePlusPlusDetector:

    def __init__(self, model_path):

        if torch.cuda.is_available():

            self.device = torch.device(

                "cuda:0"

            )

        else:

            self.device = torch.device(

                "cpu"

            )

        print("=" * 72)

        print("Loading Lastde++ proxy model")

        print(f"Model path         : {model_path}")

        print(f"Device             : {self.device}")

        print(f"N samples          : {N_SAMPLES}")

        print(f"Embedding size     : {EMBED_SIZE}")

        print(f"Epsilon            : {EPSILON_MULTIPLIER} * n")

        print(f"Tau prime          : {TAU_PRIME}")

        print(f"Aggregation        : Std")

        print(f"Max length         : {MAX_LENGTH}")

        print("=" * 72)

        self.tokenizer = (

            AutoTokenizer.from_pretrained(

                model_path,

                trust_remote_code=True

            )

        )

        model_kwargs = {

            "trust_remote_code": True,

            "low_cpu_mem_usage": True

        }

        if self.device.type == "cuda":

            model_kwargs[

                "torch_dtype"

            ] = MODEL_DTYPE

        self.model = (

            AutoModelForCausalLM

            .from_pretrained(

                model_path,

                **model_kwargs

            )

            .to(self.device)

        )

        self.model.eval()

        self.model.requires_grad_(False)

        if self.tokenizer.pad_token_id is None:

            if (

                self.tokenizer.eos_token_id

                is not None

            ):

                self.tokenizer.pad_token = (

                    self.tokenizer.eos_token

                )

            else:

                raise ValueError(

                    "Tokenizer has neither "

                    "pad_token_id nor eos_token_id."

                )

        print(

            "Model loaded successfully.\n"

        )

    # Sampling

    @torch.inference_mode()

    def get_samples(

        self,

        logits

    ):

        log_probs = torch.log_softmax(

            logits.float(),

            dim=-1

        )

        distribution = (

            torch.distributions.Categorical(

                logits=log_probs

            )

        )

        samples = (

            distribution

            .sample(

                [N_SAMPLES]

            )

            .permute(

                1,

                2,

                0

            )

        )

        return samples

    # Token log-likelihood

    @torch.inference_mode()

    def get_likelihood(

        self,

        logits,

        labels

    ):

        if (

            labels.ndim

            ==

            logits.ndim - 1

        ):

            labels = (

                labels.unsqueeze(-1)

            )

        log_probs = torch.log_softmax(

            logits.float(),

            dim=-1

        )

        return log_probs.gather(

            dim=-1,

            index=labels.long()

        )

    # Lastde

    @torch.inference_mode()

    def get_lastde(

        self,

        log_likelihood

    ):

        n = (

            log_likelihood

            .shape[1]

        )

        epsilon = int(

            EPSILON_MULTIPLIER

            *

            n

        )

        mean_ll = (

            log_likelihood

            .mean(

                dim=1

            )

            .squeeze(0)

        )

        agg_mde, effective_tau = (

            get_tau_multiscale_de(

                ori_data=log_likelihood,

                embed_size=EMBED_SIZE,

                epsilon=epsilon,

                tau_prime=TAU_PRIME

            )

        )

        if torch.any(

            torch.abs(

                agg_mde

            ) < 1e-12

        ):

            agg_mde = torch.where(

                torch.abs(

                    agg_mde

                ) < 1e-12,

                torch.full_like(

                    agg_mde,

                    1e-12

                ),

                agg_mde

            )

        lastde = (

            mean_ll

            /

            agg_mde

        )

        return (

            lastde,

            effective_tau

        )

    # Lastde++ discrepancy

    @torch.inference_mode()

    def sampling_discrepancy(

        self,

        logits,

        labels

    ):

        samples = self.get_samples(

            logits

        )

        observed_ll = (

            self.get_likelihood(

                logits,

                labels

            )

        )

        sampled_ll = (

            self.get_likelihood(

                logits,

                samples

            )

        )

        observed_lastde, _ = (

            self.get_lastde(

                observed_ll

            )

        )

        sampled_lastde, _ = (

            self.get_lastde(

                sampled_ll

            )

        )

        mu_tilde = (

            sampled_lastde

            .mean()

        )

        sigma_tilde = (

            sampled_lastde

            .std()

        )

        if (

            not torch.isfinite(

                sigma_tilde

            )

            or

            sigma_tilde.abs()

            < 1e-12

        ):

            raise RuntimeError(

                "Sampled Lastde standard "

                "deviation is zero/non-finite."

            )

        discrepancy = (

            observed_lastde.squeeze()

            -

            mu_tilde

        ) / sigma_tilde

        return float(

            discrepancy.item()

        )

    # Score one text

    @torch.inference_mode()

    def score(

        self,

        text

    ):

        encoded = self.tokenizer(

            text,

            return_tensors="pt",

            return_attention_mask=True,

            return_token_type_ids=False,

            truncation=True,

            max_length=MAX_LENGTH

        )

        encoded = {

            k: v.to(

                self.device

            )

            for k, v

            in encoded.items()

        }

        input_ids = (

            encoded[

                "input_ids"

            ]

        )

        if (

            input_ids.shape[1]

            <= EMBED_SIZE + 2

        ):

            raise ValueError(

                f"Text is too short after "

                f"tokenization: "

                f"{input_ids.shape[1]} tokens."

            )

        labels = (

            input_ids[

                :,

                1:

            ]

        )

        outputs = self.model(

            **encoded,

            use_cache=False,

            return_dict=True

        )

        logits = (

            outputs.logits[

                :,

                :-1,

                :

            ]

        )

        return self.sampling_discrepancy(

            logits=logits,

            labels=labels

        )

# Evaluate one file

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

            if math.isfinite(

                score

            ):

                y_true.append(

                    item["label"]

                )

                y_score.append(

                    score

                )

        except Exception as e:

            print(

                f"\nFailed sample in "

                f"{json_path.name}: "

                f"{e}"

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

        or

        n_rewritten == 0

    ):

        raise RuntimeError(

            f"No valid scores for both "

            f"classes. "

            f"original={n_original}, "

            f"rewritten={n_rewritten}"

        )

    raw_auroc = roc_auc_score(

        y_true,

        y_score

    )

    flipped_auroc = roc_auc_score(

        y_true,

        [-x for x in y_score]

    )

    return (

        float(raw_auroc),

        float(flipped_auroc)

    )

# Find compatible dataset files

def find_dataset_files(

    data_dir

):

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

        data_dir.glob(

            "*.json"

        )

    ):

        if (

            path.resolve()

            ==

            result_path

        ):

            continue

        lower_name = (

            path.name.lower()

        )

        if (

            lower_name.endswith(

                "_lastdepp_scores.json"

            )

            or

            lower_name.endswith(

                "_lastdepp_summary.json"

            )

            or

            lower_name

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

                isinstance(

                    obj,

                    dict

                )

                and

                "original"

                in obj

                and

                "rewritten"

                in obj

                and

                isinstance(

                    obj["original"],

                    list

                )

                and

                isinstance(

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

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(

            SEED

        )

    data_dir = Path(

        DATA_PATH

    )

    dataset_files = (

        find_dataset_files(

            data_dir

        )

    )

    if len(

        dataset_files

    ) == 0:

        raise RuntimeError(

            f"No compatible JSON datasets "

            f"found under:\n"

            f"{DATA_PATH}"

        )

    print("=" * 72)

    print(

        f"Found {len(dataset_files)} "

        f"compatible dataset files:"

    )

    for path in dataset_files:

        print(

            f"  - {path.name}"

        )

    print("=" * 72)

    # Load model ONCE

    detector = (

        LastdePlusPlusDetector(

            MODEL_PATH

        )

    )

    all_results = []

    # Evaluate all datasets

    for file_index, json_path in enumerate(

        dataset_files,

        start=1

    ):

        print()

        print(

            f"[{file_index}/"

            f"{len(dataset_files)}] "

            f"Processing: "

            f"{json_path.name}"

        )

        try:

            raw_auroc, flipped_auroc = (

                evaluate_one_file(

                    detector,

                    json_path

                )

            )

            all_results.append({

                "text_name":

                    json_path.name,

                "auroc":

                    raw_auroc

            })

            print(

                f"{json_path.name} -> "

                f"AUROC(raw) = "

                f"{raw_auroc:.6f}"

            )

            print(

                f"{json_path.name} -> "

                f"AUROC(flipped, diagnostic) = "

                f"{flipped_auroc:.6f}"

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

    # Save ONE JSON

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

    print(

        "All datasets finished."

    )

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

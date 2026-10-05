#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Standalone ProSSD tokenizer-structure adaptation for batch evaluation."""
import gc
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# USER CONFIGURATION

TRAIN_DATA_PATH = "/path/to/train.json"

TEST_DATA_DIR = "/path/to/test_data"

# Change this only if your local RoBERTa-large folder has another name.

ROBERTA_MODEL_PATH = "/path/to/roberta-large"

GPU_ID = 0

RESULT_FILE_NAME = "prossd_tokenizer_all_results.json"

MODEL_FILE_NAME = "prossd_tokenizer_model.pkl"

CACHE_DIR = str(

    Path(TRAIN_DATA_PATH).with_name("prossd_tokenizer_cache")

)

MODEL_SAVE_PATH = str(

    Path(TRAIN_DATA_PATH).with_name(MODEL_FILE_NAME)

)

# PAPER SETTINGS

PROJECTION_DIM = 4

WINDOW_SIZE = 2

STRIDE = 1

SEED = 42

# Paper comparative setting: 1400 HWT and MGT samples.

PAPER_MAX_PER_CLASS = 1400

# RoBERTa has a 512-token limit.

MAX_SUBWORD_LENGTH = 512

# Conservative batch size for RoBERTa-large.

EMBED_BATCH_SIZE = 4

# Numerical stabilization only; the paper does not specify a ridge value.

# It does not alter the method's objective, only prevents singular matrices.

COV_EPS = 1e-5

# A Gaussian covariance in 2k=8 dimensions needs enough observations.

# 10 is a numerical-safety cutoff for rare tokenizer-structure bigrams.

MIN_STRUCTURE_COUNT_PER_CLASS = 10

# Use BF16 on CUDA if supported; otherwise FP32.

USE_BF16 = True

# REPRODUCIBILITY

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

# DATA

def load_json_dataset(path):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:

        obj = json.load(f)

    if not isinstance(obj, dict):

        raise ValueError(f"{path}: top-level JSON must be a dict.")

    if "original" not in obj or "rewritten" not in obj:

        raise ValueError(

            f"{path}: must contain 'original' and 'rewritten'."

        )

    rows = []

    for text in obj["original"]:

        if isinstance(text, str) and text.strip():

            rows.append({

                "text": text.strip(),

                "label": 0

            })

    for text in obj["rewritten"]:

        if isinstance(text, str) and text.strip():

            rows.append({

                "text": text.strip(),

                "label": 1

            })

    n_h = sum(x["label"] == 0 for x in rows)

    n_m = sum(x["label"] == 1 for x in rows)

    if n_h == 0 or n_m == 0:

        raise ValueError(

            f"{path}: both classes required; human={n_h}, machine={n_m}"

        )

    print(

        f"{path.name}: human={n_h}, machine={n_m}, total={len(rows)}"

    )

    return rows

def paper_sample_training_rows(rows):

    """
    Use all available examples if class size <= 1400.

    If larger, sample 1400 per class using seed 42.

    """
    rng = np.random.default_rng(SEED)

    selected = []

    for label in (0, 1):

        class_rows = [

            x for x in rows

            if x["label"] == label

        ]

        if len(class_rows) > PAPER_MAX_PER_CLASS:

            idx = rng.choice(

                len(class_rows),

                size=PAPER_MAX_PER_CLASS,

                replace=False

            )

            class_rows = [

                class_rows[i]

                for i in sorted(idx.tolist())

            ]

        selected.extend(class_rows)

    rng.shuffle(selected)

    print(

        "Training samples actually used: "

        f"human={sum(x['label']==0 for x in selected)}, "

        f"machine={sum(x['label']==1 for x in selected)}, "

        f"total={len(selected)}"

    )

    return selected

def find_dataset_files(data_dir):

    data_dir = Path(data_dir)

    if not data_dir.exists():

        raise FileNotFoundError(

            f"TEST_DATA_DIR does not exist: {data_dir}"

        )

    result_path = (

        data_dir / RESULT_FILE_NAME

    ).resolve()

    candidates = []

    for path in sorted(data_dir.glob("*.json")):

        if path.resolve() == result_path:

            continue

        lower = path.name.lower()

        if (

            lower.endswith("_scores.json")

            or lower.endswith("_summary.json")

            or lower in {

                "nts_all_results.json",

                "exons_all_results.json",

                "lastdepp_all_results.json",

                "mrange_all_results.json",

                "sentra_all_results.json",

                RESULT_FILE_NAME.lower(),

            }

        ):

            continue

        try:

            with open(path, "r", encoding="utf-8") as f:

                obj = json.load(f)

            if (

                isinstance(obj, dict)

                and isinstance(obj.get("original"), list)

                and isinstance(obj.get("rewritten"), list)

            ):

                candidates.append(path)

        except Exception:

            continue

    return candidates

# TEXT FEATURE EXTRACTOR

class SemanticStructuralExtractor:

    """
    Tokenizer-level adaptation of ProSSD.

    Instead of spaCy POS tags, each RoBERTa BPE token is assigned one

    coarse structural category. Adjacent category pairs are then used

    as the conditioning structure pi_t.

    Categories:

        WORD_START : RoBERTa token starts with Ġ and contains letters

        WORD_CONT  : alphabetic continuation/subword

        NUM        : token contains digits and no letters

        PUNCT      : punctuation-only token

        MIXED      : alphanumeric/mixed token

        OTHER      : anything else

    """
    def __init__(self, model_path, gpu_id):

        if not torch.cuda.is_available():

            raise RuntimeError(

                "CUDA is required for RoBERTa-large extraction."

            )

        self.device = torch.device(

            f"cuda:{gpu_id}"

        )

        print("=" * 72)

        print("Loading ProSSD tokenizer-level feature extractor")

        print(f"RoBERTa-large : {model_path}")

        print(f"GPU           : {gpu_id} -> {self.device}")

        print("Structure     : tokenizer-level BPE categories")

        print("=" * 72)

        self.tokenizer = AutoTokenizer.from_pretrained(

            model_path,

            use_fast=True,

            trust_remote_code=True

        )

        dtype = (

            torch.bfloat16

            if USE_BF16 and torch.cuda.is_bf16_supported()

            else torch.float32

        )

        self.encoder = AutoModel.from_pretrained(

            model_path,

            torch_dtype=dtype,

            low_cpu_mem_usage=True,

            trust_remote_code=True,

        ).to(self.device)

        self.encoder.eval()

        self.encoder.requires_grad_(False)

        hidden_size = int(

            getattr(

                self.encoder.config,

                "hidden_size",

                -1

            )

        )

        if hidden_size != 1024:

            print(

                f"WARNING: paper uses RoBERTa-large 1024-d final hidden "

                f"states, but loaded model hidden_size={hidden_size}."

            )

        self.hidden_size = hidden_size

    @staticmethod

    def token_structure(token):

        """
        Convert one RoBERTa BPE token string to a coarse tokenizer-level

        structural category.

        """
        # RoBERTa byte-level BPE uses Ġ to mark many word starts.

        is_word_start = token.startswith("Ġ")

        clean = token.lstrip("Ġ")

        if clean == "":

            return "OTHER"

        has_alpha = any(ch.isalpha() for ch in clean)

        has_digit = any(ch.isdigit() for ch in clean)

        has_punct = any(

            (not ch.isalnum()) and (not ch.isspace())

            for ch in clean

        )

        if has_alpha and not has_digit and not has_punct:

            return "WORD_START" if is_word_start else "WORD_CONT"

        if has_digit and not has_alpha and not has_punct:

            return "NUM"

        if has_punct and not has_alpha and not has_digit:

            return "PUNCT"

        if has_alpha or has_digit:

            return "MIXED"

        return "OTHER"

    @torch.inference_mode()

    def extract_one(self, text):

        encoded = self.tokenizer(

            text,

            truncation=True,

            max_length=MAX_SUBWORD_LENGTH,

            return_tensors="pt",

            return_attention_mask=True,

            return_special_tokens_mask=True,

            add_special_tokens=True,

        )

        input_ids_cpu = encoded["input_ids"][0]

        attention_mask_cpu = encoded["attention_mask"][0].bool()

        special_mask_cpu = encoded["special_tokens_mask"][0].bool()

        # Keep only real, non-special tokens.

        valid_mask_cpu = (

            attention_mask_cpu

            & (~special_mask_cpu)

        )

        valid_indices = valid_mask_cpu.nonzero(

            as_tuple=False

        ).squeeze(-1)

        if valid_indices.numel() < WINDOW_SIZE:

            return None

        # Do not pass special_tokens_mask into the encoder.

        encoded_gpu = {

            k: v.to(self.device)

            for k, v in encoded.items()

            if k != "special_tokens_mask"

        }

        outputs = self.encoder(

            **encoded_gpu,

            output_hidden_states=True,

            return_dict=True

        )

        # Final-layer token representations.

        hidden = outputs.hidden_states[-1][0].float()

        embeddings = hidden[

            valid_indices.to(self.device)

        ].cpu().numpy().astype(

            np.float32,

            copy=False

        )

        token_ids = input_ids_cpu[

            valid_indices

        ].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(

            token_ids

        )

        structures = [

            self.token_structure(tok)

            for tok in tokens

        ]

        del outputs, hidden, encoded_gpu

        return {

            "embeddings": embeddings,

            "structures": structures,

            "tokens": tokens,

        }

# CACHE

def safe_cache_name(path):

    return (

        Path(path).name

        .replace("/", "_")

        .replace("\\\\", "_")

        + ".prossd_tokenizer_features.pkl"

    )

def extract_records_with_cache(

    extractor,

    rows,

    cache_path,

    description

):

    cache_path = Path(cache_path)

    if cache_path.exists():

        print(f"Loading feature cache: {cache_path}")

        with open(cache_path, "rb") as f:

            return pickle.load(f)

    features = []

    for row in tqdm(

        rows,

        desc=description

    ):

        item = extractor.extract_one(

            row["text"]

        )

        if item is None:

            features.append(None)

            continue

        item["label"] = int(

            row["label"]

        )

        features.append(item)

    payload = {

        "features": features,

        "model_path": ROBERTA_MODEL_PATH,

        "projection_dim": PROJECTION_DIM,

        "max_subword_length": MAX_SUBWORD_LENGTH,

        "seed": SEED,

    }

    cache_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(cache_path, "wb") as f:

        pickle.dump(

            payload,

            f,

            protocol=pickle.HIGHEST_PROTOCOL

        )

    print(f"Saved feature cache: {cache_path}")

    return payload

# SUPERVISED SUBSPACE PROJECTION

def fit_projection(train_features):

    """
    Paper Sec. 3.2 / Appendix C.1:

      maximize squared covariance with class labels and recursively deflate.

    The appendix explicitly connects this recursion to NIPALS PLS.

    sklearn PLSRegression uses NIPALS-style supervised latent components.

    We disable feature scaling so the construction operates on centered raw

    RoBERTa hidden states as described in the paper.

    """
    xs = []

    ys = []

    for item in train_features:

        if item is None:

            continue

        E = item["embeddings"]

        xs.append(E)

        ys.append(

            np.full(

                E.shape[0],

                item["label"],

                dtype=np.float32

            )

        )

    X = np.concatenate(

        xs,

        axis=0

    ).astype(

        np.float64,

        copy=False

    )

    y = np.concatenate(

        ys,

        axis=0

    ).reshape(-1, 1).astype(

        np.float64,

        copy=False

    )

    print()

    print("=" * 72)

    print("Learning supervised ProSSD subspace")

    print(f"Word vectors : {X.shape[0]}")

    print(f"Input dim    : {X.shape[1]}")

    print(f"Projection k : {PROJECTION_DIM}")

    print("=" * 72)

    pls = PLSRegression(

        n_components=PROJECTION_DIM,

        scale=False,

        max_iter=500,

        tol=1e-06

    )

    pls.fit(

        X,

        y

    )

    # x_rotations_ maps centered original embeddings to latent scores.

    P = pls.x_rotations_.astype(

        np.float64,

        copy=True

    )

    x_mean = pls._x_mean.astype(

        np.float64,

        copy=True

    )

    return {

        "P": P,

        "x_mean": x_mean,

    }

def project_embeddings(

    embeddings,

    projection

):

    X = embeddings.astype(

        np.float64,

        copy=False

    )

    return (

        (X - projection["x_mean"])

        @ projection["P"]

    )

# GAUSSIAN / WASSERSTEIN UTILITIES

def regularized_cov(X):

    X = np.asarray(

        X,

        dtype=np.float64

    )

    if X.shape[0] < 2:

        raise ValueError(

            "At least two samples required for covariance."

        )

    cov = np.cov(

        X,

        rowvar=False,

        ddof=1

    )

    cov = np.atleast_2d(

        cov

    ).astype(

        np.float64

    )

    cov = (

        0.5 * (cov + cov.T)

        + COV_EPS * np.eye(cov.shape[0])

    )

    return cov

def psd_sqrt(A):

    A = 0.5 * (

        A + A.T

    )

    vals, vecs = eigh(A)

    vals = np.clip(

        vals,

        a_min=0.0,

        a_max=None

    )

    return (

        vecs

        @ np.diag(np.sqrt(vals))

        @ vecs.T

    )

def gaussian_wasserstein(mu_h, cov_h, mu_m, cov_m):

    """
    Eq. 6 in the paper.

    """
    diff = mu_h - mu_m

    sqrt_h = psd_sqrt(

        cov_h

    )

    middle = (

        sqrt_h

        @ cov_m

        @ sqrt_h

    )

    sqrt_middle = psd_sqrt(

        middle

    )

    w2_sq = (

        float(diff @ diff)

        +

        float(

            np.trace(

                cov_h

                + cov_m

                - 2.0 * sqrt_middle

            )

        )

    )

    return math.sqrt(

        max(w2_sq, 0.0)

    )

def gaussian_params(X):

    X = np.asarray(

        X,

        dtype=np.float64

    )

    mu = X.mean(

        axis=0

    )

    cov = regularized_cov(

        X

    )

    precision = np.linalg.pinv(

        cov

    )

    sign, logdet = np.linalg.slogdet(

        cov

    )

    if sign <= 0:

        # COV_EPS should normally prevent this.

        logdet = math.log(

            max(

                abs(np.linalg.det(cov)),

                1e-300

            )

        )

    return {

        "mu": mu,

        "cov": cov,

        "precision": precision,

        "logdet": float(logdet),

    }

# SEMANTIC-STRUCTURAL DISTRIBUTION LIBRARY

def build_distribution_library(

    train_features,

    projection

):

    """
    Tokenizer-level adaptation of Eq. 3-6 / Algorithm 1.

    π_t = (structure_t, structure_{t+1})

    x_t = [v_t ; v_{t+1}]

    """
    grouped = {

        0: defaultdict(list),

        1: defaultdict(list),

    }

    for item in tqdm(

        train_features,

        desc="Building semantic-structural library"

    ):

        if item is None:

            continue

        V = project_embeddings(

            item["embeddings"],

            projection

        )

        structures = item["structures"]

        label = int(

            item["label"]

        )

        n = min(

            len(structures),

            V.shape[0]

        )

        for t in range(

            0,

            n - WINDOW_SIZE + 1,

            STRIDE

        ):

            # Paper's main setting is window size 2.

            if WINDOW_SIZE != 2:

                raise RuntimeError(

                    "This reproduction implements the paper's window size 2."

                )

            x_t = np.concatenate(

                [

                    V[t],

                    V[t + 1]

                ],

                axis=0

            )

            pi_t = (

                structures[t],

                structures[t + 1]

            )

            grouped[label][pi_t].append(

                x_t

            )

    common_structures = sorted(

        set(grouped[0].keys())

        & set(grouped[1].keys())

    )

    library = {}

    skipped = 0

    for pi in common_structures:

        H = np.asarray(

            grouped[0][pi],

            dtype=np.float64

        )

        M = np.asarray(

            grouped[1][pi],

            dtype=np.float64

        )

        if (

            H.shape[0] < MIN_STRUCTURE_COUNT_PER_CLASS

            or M.shape[0] < MIN_STRUCTURE_COUNT_PER_CLASS

        ):

            skipped += 1

            continue

        h_par = gaussian_params(

            H

        )

        m_par = gaussian_params(

            M

        )

        weight = gaussian_wasserstein(

            h_par["mu"],

            h_par["cov"],

            m_par["mu"],

            m_par["cov"]

        )

        if not math.isfinite(weight):

            continue

        library[pi] = {

            "human": h_par,

            "machine": m_par,

            "weight": float(weight),

            "n_human": int(H.shape[0]),

            "n_machine": int(M.shape[0]),

        }

    print()

    print("=" * 72)

    print("Distribution library built")

    print(f"Common token bigrams: {len(common_structures)}")

    print(f"Usable structures : {len(library)}")

    print(f"Skipped rare      : {skipped}")

    print("=" * 72)

    if len(library) == 0:

        raise RuntimeError(

            "No usable tokenizer structures were learned. "

            "Check the training data and tokenizer outputs."

        )

    return library

# DETECTION

def modified_mahalanobis(

    x,

    params

):

    """
    Eq. 10:

      D_M(x;mu,Sigma)

        = (x-mu)^T Sigma^{-1}(x-mu) + ln|Sigma|

    """
    delta = (

        x - params["mu"]

    )

    quad = float(

        delta

        @ params["precision"]

        @ delta

    )

    return (

        quad

        + params["logdet"]

    )

def score_feature_item(

    item,

    projection,

    library

):

    """
    Eq. 8-11:

      s_t = 0.5 * [D_H - D_M]

      S(T) = sum(w_pi * s_t) / sum(w_pi)

    Larger score => more machine-like.

    """
    if item is None:

        return float("nan")

    V = project_embeddings(

        item["embeddings"],

        projection

    )

    structures = item["structures"]

    n = min(

        len(structures),

        V.shape[0]

    )

    weighted_score = 0.0

    weight_sum = 0.0

    used = 0

    for t in range(

        0,

        n - WINDOW_SIZE + 1,

        STRIDE

    ):

        pi = (

            structures[t],

            structures[t + 1]

        )

        stats = library.get(

            pi

        )

        if stats is None:

            continue

        x_t = np.concatenate(

            [

                V[t],

                V[t + 1]

            ],

            axis=0

        )

        d_h = modified_mahalanobis(

            x_t,

            stats["human"]

        )

        d_m = modified_mahalanobis(

            x_t,

            stats["machine"]

        )

        s_t = 0.5 * (

            d_h - d_m

        )

        w = stats["weight"]

        weighted_score += (

            w * s_t

        )

        weight_sum += w

        used += 1

    if used == 0 or weight_sum <= 0:

        return float("nan")

    return float(

        weighted_score / weight_sum

    )

def evaluate_features(

    features,

    projection,

    library

):

    y_true = []

    y_score = []

    for item in features:

        if item is None:

            continue

        score = score_feature_item(

            item,

            projection,

            library

        )

        if math.isfinite(score):

            y_true.append(

                int(item["label"])

            )

            y_score.append(

                float(score)

            )

    if len(set(y_true)) != 2:

        raise RuntimeError(

            "Need valid scores from both classes for AUROC."

        )

    return float(

        roc_auc_score(

            y_true,

            y_score

        )

    ), len(y_true)

# SAVE / LOAD TRAINED ProSSD

def save_prossd_model(

    projection,

    library,

    train_count

):

    payload = {

        "projection": projection,

        "library": library,

        "train_count": train_count,

        "paper_config": {

            "semantic_model": ROBERTA_MODEL_PATH,

            "semantic_layer": "final",

            "semantic_dim_expected": 1024,

            "structure_source": "roberta_tokenizer_level_categories",

            "structure_categories": [

                "WORD_START",

                "WORD_CONT",

                "NUM",

                "PUNCT",

                "MIXED",

                "OTHER",

            ],

            "projection_dim": PROJECTION_DIM,

            "window_size": WINDOW_SIZE,

            "stride": STRIDE,

            "seed": SEED,

            "paper_max_per_class": PAPER_MAX_PER_CLASS,

        },

        "implementation_safety": {

            "cov_eps": COV_EPS,

            "min_structure_count_per_class":

                MIN_STRUCTURE_COUNT_PER_CLASS,

            "representation_level": "roberta_bpe_token",

            "paper_deviation":

                "spaCy POS bigrams replaced by tokenizer-level structural bigrams",

        }

    }

    with open(

        MODEL_SAVE_PATH,

        "wb"

    ) as f:

        pickle.dump(

            payload,

            f,

            protocol=pickle.HIGHEST_PROTOCOL

        )

    print(

        f"Saved trained ProSSD statistics to:\n"

        f"{MODEL_SAVE_PATH}"

    )

# MAIN

def main():

    set_seed(SEED)

    if not Path(TRAIN_DATA_PATH).exists():

        raise FileNotFoundError(

            f"Training data not found:\n{TRAIN_DATA_PATH}"

        )

    if not Path(TEST_DATA_DIR).exists():

        raise FileNotFoundError(

            f"Test directory not found:\n{TEST_DATA_DIR}"

        )

    if not Path(ROBERTA_MODEL_PATH).exists():

        raise FileNotFoundError(

            "RoBERTa-large local model was not found at:\n"

            f"{ROBERTA_MODEL_PATH}\n\n"

            "Edit ROBERTA_MODEL_PATH to your actual local roberta-large folder."

        )

    Path(CACHE_DIR).mkdir(

        parents=True,

        exist_ok=True

    )

    extractor = SemanticStructuralExtractor(

        ROBERTA_MODEL_PATH,

        GPU_ID

    )

    # Fit ProSSD on user's merged_data.json

    train_rows_all = load_json_dataset(

        TRAIN_DATA_PATH

    )

    train_rows = paper_sample_training_rows(

        train_rows_all

    )

    train_cache = (

        Path(CACHE_DIR)

        / "merged_data.prossd_tokenizer_features.pkl"

    )

    train_payload = extract_records_with_cache(

        extractor,

        train_rows,

        train_cache,

        "Extracting training RoBERTa/token-structure features"

    )

    train_features = train_payload[

        "features"

    ]

    projection = fit_projection(

        train_features

    )

    library = build_distribution_library(

        train_features,

        projection

    )

    save_prossd_model(

        projection,

        library,

        train_count={

            "human":

                sum(x["label"] == 0 for x in train_rows),

            "machine":

                sum(x["label"] == 1 for x in train_rows),

        }

    )

    # Evaluate every JSON under shuju/

    dataset_files = find_dataset_files(

        TEST_DATA_DIR

    )

    if len(dataset_files) == 0:

        raise RuntimeError(

            f"No compatible JSON files found under:\n"

            f"{TEST_DATA_DIR}"

        )

    print()

    print("=" * 72)

    print(

        f"Found {len(dataset_files)} test datasets"

    )

    print("=" * 72)

    all_results = []

    for file_index, json_path in enumerate(

        dataset_files,

        start=1

    ):

        print()

        print(

            f"[{file_index}/{len(dataset_files)}] "

            f"{json_path.name}"

        )

        try:

            rows = load_json_dataset(

                json_path

            )

            # Test-set features are extracted in memory only.

            # They are NOT written to disk, avoiding large per-dataset cache files.

            test_features = []

            for row in tqdm(

                rows,

                desc=f"Features {json_path.name}"

            ):

                item = extractor.extract_one(

                    row["text"]

                )

                if item is None:

                    test_features.append(None)

                    continue

                item["label"] = int(

                    row["label"]

                )

                test_features.append(

                    item

                )

            auroc, n_valid = evaluate_features(

                test_features,

                projection,

                library

            )

            # Release test features immediately after scoring.

            del test_features

            all_results.append({

                "text_name":

                    json_path.name,

                "auroc":

                    auroc,

            })

            print(

                f"{json_path.name} -> "

                f"AUROC={auroc:.6f} "

                f"(valid={n_valid})"

            )

        except Exception as e:

            all_results.append({

                "text_name":

                    json_path.name,

                "auroc":

                    None,

            })

            print(

                f"{json_path.name} -> FAILED: {e}"

            )

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # Save exactly one summary JSON

    result_path = (

        Path(TEST_DATA_DIR)

        / RESULT_FILE_NAME

    )

    with open(

        result_path,

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

    print("All ProSSD evaluations finished.")

    print(f"Result: {result_path}")

    print("=" * 72)

    for row in all_results:

        score_text = (

            "FAILED"

            if row["auroc"] is None

            else f"{row['auroc']:.6f}"

        )

        print(

            f"{row['text_name']}: "

            f"{score_text}"

        )

if __name__ == "__main__":

    main()

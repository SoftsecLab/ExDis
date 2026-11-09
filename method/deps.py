"""Dependencies and unchanged experiment configuration."""
import copy
import json
import math
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Reduce CUDA allocator fragmentation. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer


# ============================================================
# USER CONFIGURATION -- EDIT THE FIVE PATHS
# ============================================================
TRAIN_DATA_PATH = "/home/gsy/project2/merged_data.json"
TEST_DATA_PATH = "/home/gsy/project2/my_method2/shuju/writingPrompts_polish_qwen_7b.raw_data.json"
SCORING_MODEL_PATH = "/home/share/models/gpt-neo-2.7B"
PARAPHRASER_MODEL_PATH = "/home/share/models/t5-base"
# REQUIRED: path to YOUR independent validation JSON (same format as train/test).
# This file is used IN FULL; training data will NOT be split for validation.
# Example: VALIDATION_DATA_PATH = "/home/gsy/project2/my_method2/shuju/my_validation.json"
# Do NOT use your training or test dataset here.
VALIDATION_DATA_PATH = "/home/gsy/project2/my_method2/shuju/xsum_polish_gpt-4o.raw_data.json"


# ============================================================
# ORIGINAL HYPERPARAMETERS (kept unchanged)
# ============================================================
SCORING_LR = 1e-4
PREFERENCE_BETA = 2.0
PREFERENCE_MARGIN_GAMMA = 1.0
GRAD_ACCUM_STEPS = 4
EPOCHS = 10
# Validation is performed twice per epoch, after an optimizer update.
VALIDATIONS_PER_EPOCH = 2
EARLY_STOP_PATIENCE = 2  # consecutive validation checks (not epochs)
EARLY_STOP_MIN_DELTA = 0.005  # absolute AUROC improvement (0.5 percentage points)
# Fixed to match the original batch inference script and the paper experiment:
# DeltaLRP(x) = LRP(x; optimized scoring model) - LRP(x; original base model).
DELTA_LRP_DIRECTION = "optimized_minus_original"
DATANUM = 1000
SEED = 42
BATCH_SIZE = 1
PPO_EPSILON = 0.2
ENTROPY_COEF = 0.01
PARAPHRASER_LR = 5e-4
REWARD_EMA_ALPHA = 0.95
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
T5_MAX_LENGTH = 200
TOP_K = 30
TOP_P = 0.95
TEMPERATURE = 1.0
SCORING_MAX_LENGTH = 512

# Memory-safety settings. These only trade speed for lower peak VRAM.
USE_BF16 = True
ENABLE_GRADIENT_CHECKPOINTING = True
AGGRESSIVE_T5_CPU_OFFLOAD = True
LOGIT_CHUNK_SIZE = 32

# Requested perturbation range.
MIN_PERTURB_RATIO = 0.10
MAX_PERTURB_RATIO = 0.20

OUTPUT_DIR = "./etd_corrected_output11_earlystop_specified_val_v2"
SCORING_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "scoring_adapter")
PARAPHRASER_SAVE_DIR = os.path.join(OUTPUT_DIR, "paraphraser")
RESULT_JSON = os.path.join(OUTPUT_DIR, "test_results.json")


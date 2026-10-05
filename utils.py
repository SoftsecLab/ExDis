"""Runtime and model-loading utilities for ExDis."""

import gc
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import USE_BF16


def set_seed(seed):
    """Set deterministic random seeds used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device():
    """Return the CUDA device required by the original implementation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ExDis implementation.")
    return torch.device("cuda:0")


def ensure_padding_token(tokenizer, model=None):
    """Ensure that a tokenizer has a valid padding token."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))


def preferred_model_dtype(device):
    """Use BF16 on supported CUDA hardware and FP32 otherwise."""
    if USE_BF16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_causal_model(model_path, device):
    """Load a causal LM using the memory settings from the original code."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=preferred_model_dtype(device),
    )

    ensure_padding_token(tokenizer, model)
    model.to(device)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer


def move_optimizer_state(optimizer, device):
    """Move Adam optimizer-state tensors together with an offloaded model."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=False)


def cuda_cleanup():
    """Release cached CUDA memory to reduce peak reserved VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

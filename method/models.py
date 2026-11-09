"""ExDis models module."""
from .deps import *

def ensure_padding_token(tokenizer, model=None):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))


def preferred_model_dtype(device):
    if (
        USE_BF16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float32


def load_causal_model(model_path, device):
    """
    Load the causal LM in BF16 on supported CUDA hardware.
    This halves parameter VRAM relative to FP32 without changing model structure.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    dtype = preferred_model_dtype(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    ensure_padding_token(tokenizer, model)
    model.to(device)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer


def move_optimizer_state(optimizer, device):
    """
    Explicitly move optimizer state tensors with the model.
    PyTorch model.to(...) does NOT move Adam state automatically.
    """
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=False)


def cuda_cleanup():
    """
    Deliberately synchronize and release cached blocks.
    This is slower, but keeps peak reserved VRAM predictable.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()



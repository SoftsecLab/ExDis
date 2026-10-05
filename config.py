"""Default hyperparameters for ExDis.

The values below are preserved from the original low-VRAM implementation.
Dataset/model paths are intentionally not stored here; provide them through
the command-line interface in ``train.py``.
"""

# Scoring-model optimization
SCORING_LR = 1e-4
PREFERENCE_BETA = 2.0
PREFERENCE_MARGIN_GAMMA = 1.0
GRAD_ACCUM_STEPS = 4
EPOCHS = 2
DATANUM = 500
SEED = 42
BATCH_SIZE = 1

# PPO-based adaptive rewriting
PPO_EPSILON = 0.2
ENTROPY_COEF = 0.01
PARAPHRASER_LR = 5e-4
REWARD_EMA_ALPHA = 0.95

# LoRA
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Generation/scoring limits
T5_MAX_LENGTH = 200
TOP_K = 30
TOP_P = 0.95
TEMPERATURE = 1.0
SCORING_MAX_LENGTH = 512

# Low-VRAM execution
USE_BF16 = True
ENABLE_GRADIENT_CHECKPOINTING = True
AGGRESSIVE_T5_CPU_OFFLOAD = True
LOGIT_CHUNK_SIZE = 32

# Perturbation range
MIN_PERTURB_RATIO = 0.10
MAX_PERTURB_RATIO = 0.20

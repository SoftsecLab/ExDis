"""Training and checkpoint utilities for ExDis."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import (
    AGGRESSIVE_T5_CPU_OFFLOAD,
    BATCH_SIZE,
    DATANUM,
    ENABLE_GRADIENT_CHECKPOINTING,
    ENTROPY_COEF,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    LOGIT_CHUNK_SIZE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_PERTURB_RATIO,
    MIN_PERTURB_RATIO,
    PARAPHRASER_LR,
    PPO_EPSILON,
    PREFERENCE_BETA,
    PREFERENCE_MARGIN_GAMMA,
    REWARD_EMA_ALPHA,
    SCORING_LR,
    SEED,
    T5_MAX_LENGTH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    USE_BF16,
)
from .utils import preferred_model_dtype


def train(model, train_dataset):
    """Run the joint ExDis optimization loop."""
    num_train = min(DATANUM, len(train_dataset))
    subset_indices = torch.randperm(
        len(train_dataset)
    )[:num_train]
    train_subset = Subset(
        train_dataset,
        subset_indices,
    )
    loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    updates_per_epoch = math.ceil(
        len(loader) / GRAD_ACCUM_STEPS
    )
    scheduler = CosineAnnealingLR(
        model.scoring_optimizer,
        T_max=max(1, updates_per_epoch * EPOCHS),
        eta_min=0,
    )

    print("=" * 72)
    print("ExDis training")
    print(f"pairs used        : {num_train}")
    print(f"epochs            : {EPOCHS}")
    print(f"batch size        : {BATCH_SIZE}")
    print(f"grad accumulation : {GRAD_ACCUM_STEPS}")
    print(f"scoring lr        : {SCORING_LR}")
    print(f"paraphraser lr    : {PARAPHRASER_LR}")
    print(
        f"beta/gamma        : "
        f"{PREFERENCE_BETA}/{PREFERENCE_MARGIN_GAMMA}"
    )
    print(f"PPO epsilon       : {PPO_EPSILON}")
    print(f"entropy coef      : {ENTROPY_COEF}")
    print(
        f"perturb ratio     : "
        f"{MIN_PERTURB_RATIO}-{MAX_PERTURB_RATIO}"
    )
    print(
        f"BF16              : "
        f"{preferred_model_dtype(model.device) == torch.bfloat16}"
    )
    print(f"grad checkpoint   : {ENABLE_GRADIENT_CHECKPOINTING}")
    print(f"T5 CPU offload    : {AGGRESSIVE_T5_CPU_OFFLOAD}")
    print(f"logit chunk       : {LOGIT_CHUNK_SIZE}")
    print("=" * 72)

    for epoch in range(EPOCHS):
        model.sync_old_policy()
        model.set_training_modes()
        model.scoring_optimizer.zero_grad(set_to_none=True)

        scoring_losses = []
        ppo_losses = []
        rewards = []
        ratios = []
        perturbed_margins = []
        rewritten_margins = []

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        for batch_index, batch in enumerate(progress):
            human_texts = list(batch[0])
            rewritten_texts = list(batch[1])

            (
                perturbed_texts,
                masked_inputs,
                actions,
            ) = model.generate_perturbations(
                rewritten_texts
            )

            reward, _ = model.attack_reward(
                human_texts,
                perturbed_texts,
            )
            ppo_stats = model.update_paraphraser(
                masked_inputs,
                actions,
                reward,
            )

            score_output = model.scoring_preference_loss(
                human_texts,
                rewritten_texts,
                perturbed_texts,
            )
            (
                score_output["loss"]
                / GRAD_ACCUM_STEPS
            ).backward()

            should_update = (
                (batch_index + 1) % GRAD_ACCUM_STEPS == 0
                or batch_index == len(loader) - 1
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter
                        in model.scoring_model.parameters()
                        if parameter.requires_grad
                    ],
                    1.0,
                )
                model.scoring_optimizer.step()
                model.scoring_optimizer.zero_grad(
                    set_to_none=True
                )
                scheduler.step()

            score_loss = float(
                score_output["loss"].detach().item()
            )
            scoring_losses.append(score_loss)
            ppo_losses.append(ppo_stats["total_loss"])
            rewards.append(ppo_stats["reward"])
            ratios.append(ppo_stats["ratio"])
            perturbed_margins.extend(
                score_output[
                    "perturbed_margin"
                ].cpu().tolist()
            )
            rewritten_margins.extend(
                score_output[
                    "rewritten_margin"
                ].cpu().tolist()
            )

            progress.set_postfix(
                score=f"{score_loss:.4f}",
                ppo=f"{ppo_stats['total_loss']:.4f}",
                reward=f"{ppo_stats['reward']:.4f}",
                ratio=f"{ppo_stats['ratio']:.3f}",
            )

        print(
            f"Epoch {epoch + 1} scoring loss : "
            f"{np.mean(scoring_losses):.6f}"
        )
        print(
            f"Epoch {epoch + 1} PPO loss     : "
            f"{np.mean(ppo_losses):.6f}"
        )
        print(
            f"Epoch {epoch + 1} reward       : "
            f"{np.mean(rewards):.6f}"
        )
        print(
            f"Epoch {epoch + 1} PPO ratio    : "
            f"{np.mean(ratios):.6f}"
        )
        print(
            f"Epoch {epoch + 1} xp-h margin  : "
            f"{np.mean(perturbed_margins):.6f}"
        )
        print(
            f"Epoch {epoch + 1} xm-h margin  : "
            f"{np.mean(rewritten_margins):.6f}"
        )

    return model


def save_trained_models(model, output_dir):
    """Save trained adapters, paraphraser, tokenizer files, and configuration."""
    output_dir = Path(output_dir)
    scoring_adapter_dir = output_dir / "scoring_adapter"
    paraphraser_dir = output_dir / "paraphraser"

    output_dir.mkdir(parents=True, exist_ok=True)

    model.scoring_model.save_pretrained(
        scoring_adapter_dir
    )
    model.scoring_tokenizer.save_pretrained(
        scoring_adapter_dir
    )
    model.paraphraser.save_pretrained(
        paraphraser_dir
    )
    model.paraphraser_tokenizer.save_pretrained(
        paraphraser_dir
    )

    config = {
        "scoring_lr": SCORING_LR,
        "beta": PREFERENCE_BETA,
        "gamma": PREFERENCE_MARGIN_GAMMA,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS,
        "datanum": DATANUM,
        "batch_size": BATCH_SIZE,
        "ppo_epsilon": PPO_EPSILON,
        "entropy_coef": ENTROPY_COEF,
        "paraphraser_lr": PARAPHRASER_LR,
        "reward_ema_alpha": REWARD_EMA_ALPHA,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "t5_max_length": T5_MAX_LENGTH,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
        "perturb_ratio_min": MIN_PERTURB_RATIO,
        "perturb_ratio_max": MAX_PERTURB_RATIO,
        "seed": SEED,
        "use_bf16": USE_BF16,
        "gradient_checkpointing": ENABLE_GRADIENT_CHECKPOINTING,
        "aggressive_t5_cpu_offload": AGGRESSIVE_T5_CPU_OFFLOAD,
        "logit_chunk_size": LOGIT_CHUNK_SIZE,
    }

    config_path = output_dir / "training_config.json"
    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4,
        )

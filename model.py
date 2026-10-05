"""Core ExDis model containing scoring-model and adaptive-rewriter updates."""

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .config import (
    AGGRESSIVE_T5_CPU_OFFLOAD,
    ENABLE_GRADIENT_CHECKPOINTING,
    ENTROPY_COEF,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_PERTURB_RATIO,
    MIN_PERTURB_RATIO,
    PARAPHRASER_LR,
    PPO_EPSILON,
    PREFERENCE_BETA,
    REWARD_EMA_ALPHA,
    SCORING_LR,
    T5_MAX_LENGTH,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from .perturbation import (
    build_t5_sentinel_input,
    reconstruct_perturbed_text,
    strip_generated_action_ids,
)
from .scoring import (
    reference_free_preference_loss,
    sequence_mean_logprob,
)
from .utils import (
    cuda_cleanup,
    ensure_padding_token,
    load_causal_model,
    move_optimizer_state,
    preferred_model_dtype,
)


class ExDisModel(nn.Module):
    """Low-VRAM implementation of the ExDis training framework."""

    def __init__(self, scoring_model_path, paraphraser_model_path, device):
        super().__init__()
        self.device = device

        # The original scoring model is loaded lazily only for final evaluation.
        self.reference_model_path = scoring_model_path
        self.reference_model = None
        self.reference_tokenizer = None

        # Optimized scoring model theta_hat with LoRA.
        score_base, score_tokenizer = load_causal_model(
            scoring_model_path,
            device,
        )
        self.scoring_tokenizer = score_tokenizer

        if ENABLE_GRADIENT_CHECKPOINTING:
            score_base.gradient_checkpointing_enable()
            if hasattr(score_base, "enable_input_require_grads"):
                score_base.enable_input_require_grads()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            fan_in_fan_out=True,
        )
        self.scoring_model = get_peft_model(score_base, peft_config)

        if hasattr(self.scoring_model.config, "use_cache"):
            self.scoring_model.config.use_cache = False

        self.scoring_model.print_trainable_parameters()

        # T5 adaptive rewriting model.
        self.paraphraser_tokenizer = T5Tokenizer.from_pretrained(
            paraphraser_model_path
        )
        self.paraphraser = T5ForConditionalGeneration.from_pretrained(
            paraphraser_model_path,
            torch_dtype=preferred_model_dtype(device),
            low_cpu_mem_usage=True,
        ).to(device)

        ensure_padding_token(
            self.paraphraser_tokenizer,
            self.paraphraser,
        )

        if hasattr(self.paraphraser.config, "use_cache"):
            self.paraphraser.config.use_cache = False

        if ENABLE_GRADIENT_CHECKPOINTING:
            self.paraphraser.gradient_checkpointing_enable()

        # Keep the frozen PPO old policy on CPU except when it is needed.
        self.old_paraphraser = copy.deepcopy(self.paraphraser).to("cpu")
        self.old_paraphraser.eval()
        self.old_paraphraser.requires_grad_(False)

        scoring_params = [
            parameter
            for parameter in self.scoring_model.parameters()
            if parameter.requires_grad
        ]
        self.scoring_optimizer = AdamW(
            scoring_params,
            lr=SCORING_LR,
        )
        self.paraphraser_optimizer = AdamW(
            self.paraphraser.parameters(),
            lr=PARAPHRASER_LR,
        )

        self.reward_mean = torch.tensor(0.0, device=device)
        self.reward_var = torch.tensor(1.0, device=device)

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu"),
            )
            cuda_cleanup()

    def set_training_modes(self):
        """Set train/eval modes required during joint optimization."""
        self.scoring_model.train()
        self.paraphraser.train()
        if self.reference_model is not None:
            self.reference_model.eval()
        self.old_paraphraser.eval()

    def set_eval_modes(self):
        """Set all currently loaded components to evaluation mode."""
        self.scoring_model.eval()
        self.paraphraser.eval()
        if self.reference_model is not None:
            self.reference_model.eval()
        self.old_paraphraser.eval()

    @torch.no_grad()
    def sync_old_policy(self):
        """Synchronize the frozen PPO old policy with the current paraphraser."""
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu"),
            )
            cuda_cleanup()

        self.old_paraphraser.load_state_dict(
            self.paraphraser.state_dict()
        )
        self.old_paraphraser.to("cpu")
        self.old_paraphraser.eval()
        self.old_paraphraser.requires_grad_(False)

    @torch.no_grad()
    def generate_perturbations(self, rewritten_texts):
        """Generate token-level adaptive perturbations using the old T5 policy."""
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to(self.device)
            cuda_cleanup()

        perturbed_texts = []
        masked_inputs = []
        actions = []

        for text in rewritten_texts:
            ratio = random.uniform(
                MIN_PERTURB_RATIO,
                MAX_PERTURB_RATIO,
            )
            mask_info = build_t5_sentinel_input(
                self.paraphraser_tokenizer,
                text,
                ratio,
            )

            if mask_info is None:
                eos_token_id = self.paraphraser_tokenizer.eos_token_id
                perturbed_texts.append(text)
                masked_inputs.append(text)
                actions.append([
                    eos_token_id if eos_token_id is not None else 0
                ])
                continue

            encoded = self.paraphraser_tokenizer(
                mask_info["masked_text"],
                return_tensors="pt",
                truncation=True,
                max_length=T5_MAX_LENGTH,
            ).to(self.device)

            generated = self.old_paraphraser.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                top_k=TOP_K,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_length=T5_MAX_LENGTH,
                pad_token_id=self.paraphraser_tokenizer.pad_token_id,
                eos_token_id=self.paraphraser_tokenizer.eos_token_id,
            )
            action_ids = strip_generated_action_ids(
                self.paraphraser_tokenizer,
                generated[0],
            )
            perturbed = reconstruct_perturbed_text(
                self.paraphraser_tokenizer,
                mask_info,
                action_ids,
            )

            if not perturbed:
                perturbed = text

            perturbed_texts.append(perturbed)
            masked_inputs.append(mask_info["masked_text"])
            actions.append(action_ids)

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to("cpu")
            cuda_cleanup()

        return perturbed_texts, masked_inputs, actions

    @torch.no_grad()
    def attack_reward(self, human_texts, perturbed_texts):
        """Compute the reward that drives adaptive rewriting."""
        cuda_cleanup()

        human_logp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            human_texts,
            self.device,
            require_grad=False,
        )
        perturbed_logp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            perturbed_texts,
            self.device,
            require_grad=False,
        )

        discrepancy = perturbed_logp - human_logp
        reward = 1.0 - torch.sigmoid(
            PREFERENCE_BETA * discrepancy
        )
        return reward.detach(), discrepancy.detach()

    def _pad_actions(self, actions):
        """Pad generated action sequences for teacher-forced PPO updates."""
        max_length = min(
            max(len(action) for action in actions),
            T5_MAX_LENGTH,
        )
        labels = torch.full(
            (len(actions), max_length),
            -100,
            dtype=torch.long,
            device=self.device,
        )

        for index, ids in enumerate(actions):
            ids = ids[:max_length]
            labels[index, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            )

        return labels

    def update_paraphraser(self, masked_inputs, actions, rewards):
        """Perform one PPO-style update of the adaptive rewriting model."""
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to(self.device)
            move_optimizer_state(
                self.paraphraser_optimizer,
                self.device,
            )
            self.old_paraphraser.to(self.device)
            cuda_cleanup()

        inputs = self.paraphraser_tokenizer(
            list(masked_inputs),
            padding=True,
            truncation=True,
            max_length=T5_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        labels = self._pad_actions(actions)
        valid = labels != -100
        gather_labels = labels.masked_fill(~valid, 0)

        with torch.no_grad():
            old_output = self.old_paraphraser(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            old_log_probs = F.log_softmax(
                old_output.logits.float(),
                dim=-1,
            )
            old_token_logp = old_log_probs.gather(
                -1,
                gather_labels.unsqueeze(-1),
            ).squeeze(-1)

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to("cpu")
            del old_output, old_log_probs
            cuda_cleanup()

        new_output = self.paraphraser(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        new_log_probs = F.log_softmax(
            new_output.logits.float(),
            dim=-1,
        )
        new_token_logp = new_log_probs.gather(
            -1,
            gather_labels.unsqueeze(-1),
        ).squeeze(-1)

        log_ratio = (
            new_token_logp - old_token_logp
        ).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)

        # Normalize using previous running statistics so batch_size=1 works.
        previous_std = torch.sqrt(
            self.reward_var.clamp_min(1e-6)
        )
        advantages = (
            (rewards - self.reward_mean)
            / (previous_std + 1e-8)
        ).detach()
        token_advantage = advantages.unsqueeze(1).expand_as(ratio)

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - PPO_EPSILON,
            1.0 + PPO_EPSILON,
        )
        surrogate = torch.minimum(
            ratio * token_advantage,
            clipped_ratio * token_advantage,
        )
        policy_loss = (
            (-(surrogate) * valid).sum()
            / valid.sum().clamp_min(1)
        )

        probabilities = torch.softmax(
            new_output.logits.float(),
            dim=-1,
        )
        token_entropy = -(
            probabilities * new_log_probs
        ).sum(-1)
        entropy = (
            (token_entropy * valid).sum()
            / valid.sum().clamp_min(1)
        )
        total_loss = policy_loss - ENTROPY_COEF * entropy

        self.paraphraser_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.paraphraser.parameters(),
            1.0,
        )
        self.paraphraser_optimizer.step()

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu"),
            )
            del (
                new_output,
                new_log_probs,
                new_token_logp,
                probabilities,
                token_entropy,
            )
            cuda_cleanup()

        # Update EMA after constructing the current advantages.
        with torch.no_grad():
            batch_mean = rewards.mean()
            batch_var = rewards.var(unbiased=False)
            old_mean = self.reward_mean.clone()

            new_mean = (
                REWARD_EMA_ALPHA * old_mean
                + (1.0 - REWARD_EMA_ALPHA) * batch_mean
            )
            shift = batch_mean - old_mean
            new_var = (
                REWARD_EMA_ALPHA
                * (self.reward_var + shift.square())
                + (1.0 - REWARD_EMA_ALPHA) * batch_var
            )
            self.reward_mean.copy_(new_mean)
            self.reward_var.copy_(
                new_var.clamp_min(1e-6)
            )

        return {
            "total_loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "reward": float(rewards.mean().item()),
            "ratio": float(
                ratio[valid].mean().detach().item()
            ),
        }

    def scoring_preference_loss(
        self,
        human_texts,
        rewritten_texts,
        perturbed_texts,
    ):
        """Compute ExDis preference loss for rewritten and perturbed samples."""
        human_twice = list(human_texts) + list(human_texts)
        machine_union = (
            list(perturbed_texts)
            + list(rewritten_texts)
        )

        human_logp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            human_twice,
            self.device,
            require_grad=True,
        )
        machine_logp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            machine_union,
            self.device,
            require_grad=True,
        )

        loss, margins = reference_free_preference_loss(
            machine_logp,
            human_logp,
        )
        batch_size = len(human_texts)

        return {
            "loss": loss,
            "perturbed_margin": margins[:batch_size].detach(),
            "rewritten_margin": margins[batch_size:].detach(),
        }

    def release_training_only_models(self):
        """Move training-only T5 models to CPU before dual-LRP evaluation."""
        try:
            self.paraphraser.to("cpu")
            self.old_paraphraser.to("cpu")
        except Exception:
            pass
        cuda_cleanup()

    def load_reference_for_evaluation(self):
        """Lazy-load the original scoring model for final evaluation."""
        if self.reference_model is None:
            (
                self.reference_model,
                self.reference_tokenizer,
            ) = load_causal_model(
                self.reference_model_path,
                self.device,
            )
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)

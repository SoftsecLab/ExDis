"""ExDis network module."""
from .deps import *

def reference_free_preference_loss(machine_logp, human_logp):
    margin = machine_logp - human_logp
    loss = -F.logsigmoid(
        PREFERENCE_BETA * (margin - PREFERENCE_MARGIN_GAMMA)
    ).mean()
    return loss, margin


class CorrectedETD(nn.Module):
    def __init__(self, scoring_model_path, paraphraser_model_path, device):
        from .models import cuda_cleanup
        from .models import ensure_padding_token
        from .models import load_causal_model
        from .models import move_optimizer_state
        from .models import preferred_model_dtype
        super().__init__()
        self.device = device

        # Keep only the PATH to the original/reference model during training.
        # The reference model is not needed until final dual-LRP evaluation,
        # so lazy loading saves one full GPT-Neo copy of VRAM during training.
        self.reference_model_path = scoring_model_path
        self.reference_model = None
        self.reference_tokenizer = None

        # Optimized detector theta_hat with LoRA.
        score_base, score_tok = load_causal_model(
            scoring_model_path,
            device
        )
        self.scoring_tokenizer = score_tok

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

        # T5 attack model.
        self.paraphraser_tokenizer = T5Tokenizer.from_pretrained(
            paraphraser_model_path
        )

        t5_dtype = preferred_model_dtype(device)

        self.paraphraser = T5ForConditionalGeneration.from_pretrained(
            paraphraser_model_path,
            torch_dtype=t5_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        ensure_padding_token(
            self.paraphraser_tokenizer,
            self.paraphraser
        )

        if hasattr(self.paraphraser.config, "use_cache"):
            self.paraphraser.config.use_cache = False

        if ENABLE_GRADIENT_CHECKPOINTING:
            self.paraphraser.gradient_checkpointing_enable()

        # Frozen old-policy snapshot is kept on CPU except during rollout.
        # This preserves the corrected PPO behavior without permanently
        # occupying VRAM with a second T5-base.
        self.old_paraphraser = copy.deepcopy(
            self.paraphraser
        ).to("cpu")

        self.old_paraphraser.eval()
        self.old_paraphraser.requires_grad_(False)

        scoring_params = [p for p in self.scoring_model.parameters() if p.requires_grad]
        self.scoring_optimizer = AdamW(scoring_params, lr=SCORING_LR)
        self.paraphraser_optimizer = AdamW(
            self.paraphraser.parameters(), lr=PARAPHRASER_LR
        )

        self.reward_mean = torch.tensor(0.0, device=device)
        self.reward_var = torch.tensor(1.0, device=device)

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu")
            )
            cuda_cleanup()

    def set_training_modes(self):
        self.scoring_model.train()
        self.paraphraser.train()
        if self.reference_model is not None:
            self.reference_model.eval()
        self.old_paraphraser.eval()

    def set_eval_modes(self):
        self.scoring_model.eval()
        self.paraphraser.eval()
        if self.reference_model is not None:
            self.reference_model.eval()
        self.old_paraphraser.eval()

    @torch.no_grad()
    def sync_old_policy(self):
        # Both policies are synchronized on CPU to avoid transient double-T5 VRAM.
        from .models import cuda_cleanup
        from .models import move_optimizer_state
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu")
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
        # Slow but safe: only old T5 is placed on GPU for rollout generation.
        from .rewriting import build_t5_sentinel_input
        from .models import cuda_cleanup
        from .rewriting import reconstruct_perturbed_text
        from .rewriting import strip_generated_action_ids
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to(self.device)
            cuda_cleanup()

        perturbed_texts = []
        masked_inputs = []
        actions = []
        for text in rewritten_texts:
            ratio = random.uniform(MIN_PERTURB_RATIO, MAX_PERTURB_RATIO)
            info = build_t5_sentinel_input(self.paraphraser_tokenizer, text, ratio)
            if info is None:
                eos = self.paraphraser_tokenizer.eos_token_id
                perturbed_texts.append(text)
                masked_inputs.append(text)
                actions.append([eos if eos is not None else 0])
                continue
            enc = self.paraphraser_tokenizer(
                info["masked_text"],
                return_tensors="pt",
                truncation=True,
                max_length=T5_MAX_LENGTH,
            ).to(self.device)
            generated = self.old_paraphraser.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                do_sample=True,
                top_k=TOP_K,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_length=T5_MAX_LENGTH,
                pad_token_id=self.paraphraser_tokenizer.pad_token_id,
                eos_token_id=self.paraphraser_tokenizer.eos_token_id,
            )
            action_ids = strip_generated_action_ids(
                self.paraphraser_tokenizer, generated[0]
            )
            perturbed = reconstruct_perturbed_text(
                self.paraphraser_tokenizer, info, action_ids
            )
            if not perturbed:
                perturbed = text
            perturbed_texts.append(perturbed)
            masked_inputs.append(info["masked_text"])
            actions.append(action_ids)
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to("cpu")
            cuda_cleanup()

        return perturbed_texts, masked_inputs, actions

    @torch.no_grad()
    def attack_reward(self, human_texts, perturbed_texts):
        from .models import cuda_cleanup
        from .scoring import sequence_mean_logprob
        cuda_cleanup()

        human_lp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            human_texts,
            self.device,
            require_grad=False,
        )
        pert_lp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            perturbed_texts,
            self.device,
            require_grad=False,
        )
        discrepancy = pert_lp - human_lp
        reward = 1.0 - torch.sigmoid(PREFERENCE_BETA * discrepancy)
        return reward.detach(), discrepancy.detach()

    def _pad_actions(self, actions):
        max_len = min(max(len(x) for x in actions), T5_MAX_LENGTH)
        labels = torch.full(
            (len(actions), max_len),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        for i, ids in enumerate(actions):
            ids = ids[:max_len]
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return labels

    def update_paraphraser(self, masked_inputs, actions, rewards):
        # Slow but low-VRAM: current T5 and its Adam state are brought to GPU
        # only for the policy update.
        from .models import cuda_cleanup
        from .models import move_optimizer_state
        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to(self.device)
            move_optimizer_state(
                self.paraphraser_optimizer,
                self.device
            )

            # old policy is needed only for a no-grad teacher-forced pass.
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
            old_out = self.old_paraphraser(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            old_log_probs = F.log_softmax(old_out.logits.float(), dim=-1)
            old_token_lp = old_log_probs.gather(
                -1, gather_labels.unsqueeze(-1)
            ).squeeze(-1)

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.old_paraphraser.to("cpu")
            del old_out, old_log_probs
            cuda_cleanup()

        new_out = self.paraphraser(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        new_log_probs = F.log_softmax(new_out.logits.float(), dim=-1)
        new_token_lp = new_log_probs.gather(
            -1, gather_labels.unsqueeze(-1)
        ).squeeze(-1)

        log_ratio = (new_token_lp - old_token_lp).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)

        # Normalize with previous running statistics so batch_size=1 still works.
        previous_std = torch.sqrt(self.reward_var.clamp_min(1e-6))
        advantages = ((rewards - self.reward_mean) / (previous_std + 1e-8)).detach()
        token_adv = advantages.unsqueeze(1).expand_as(ratio)

        clipped_ratio = torch.clamp(
            ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON
        )
        surrogate = torch.minimum(ratio * token_adv, clipped_ratio * token_adv)
        policy_loss = (-(surrogate) * valid).sum() / valid.sum().clamp_min(1)

        probs = torch.softmax(new_out.logits.float(), dim=-1)
        token_entropy = -(probs * new_log_probs).sum(-1)
        entropy = (token_entropy * valid).sum() / valid.sum().clamp_min(1)
        total_loss = policy_loss - ENTROPY_COEF * entropy

        self.paraphraser_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.paraphraser.parameters(), 1.0)
        self.paraphraser_optimizer.step()

        if AGGRESSIVE_T5_CPU_OFFLOAD:
            self.paraphraser.to("cpu")
            move_optimizer_state(
                self.paraphraser_optimizer,
                torch.device("cpu")
            )
            del new_out, new_log_probs, new_token_lp, probs, token_entropy
            cuda_cleanup()

        # Update EMA only AFTER advantage construction.
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
                REWARD_EMA_ALPHA * (self.reward_var + shift.square())
                + (1.0 - REWARD_EMA_ALPHA) * batch_var
            )
            self.reward_mean.copy_(new_mean)
            self.reward_var.copy_(new_var.clamp_min(1e-6))

        return {
            "total_loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "reward": float(rewards.mean().item()),
            "ratio": float(ratio[valid].mean().detach().item()),
        }

    def scoring_preference_loss(self, human_texts, rewritten_texts, perturbed_texts):
        # Both xp and xm are machine-rewritten positives.
        from .scoring import sequence_mean_logprob
        human_twice = list(human_texts) + list(human_texts)
        machine_union = list(perturbed_texts) + list(rewritten_texts)

        human_lp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            human_twice,
            self.device,
            require_grad=True,
        )
        machine_lp = sequence_mean_logprob(
            self.scoring_model,
            self.scoring_tokenizer,
            machine_union,
            self.device,
            require_grad=True,
        )
        loss, margins = reference_free_preference_loss(machine_lp, human_lp)
        b = len(human_texts)
        return {
            "loss": loss,
            "perturbed_margin": margins[:b].detach(),
            "rewritten_margin": margins[b:].detach(),
        }

    def release_training_only_models(self):
        from .models import cuda_cleanup
        """
        Before evaluation, release T5 copies and optimizer states from GPU/CPU refs
        as much as possible. They are no longer needed for dual-model testing.
        """
        try:
            self.paraphraser.to("cpu")
            self.old_paraphraser.to("cpu")
        except Exception:
            pass

        cuda_cleanup()

    def load_reference_for_evaluation(self):
        from .models import load_causal_model
        if self.reference_model is None:
            (
                self.reference_model,
                self.reference_tokenizer
            ) = load_causal_model(
                self.reference_model_path,
                self.device
            )

            self.reference_model.eval()
            self.reference_model.requires_grad_(False)



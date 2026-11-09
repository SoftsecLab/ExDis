"""ExDis training module."""
from .deps import *

def _snapshot_trainable_adapter(model):
    """Keep only LoRA trainable tensors in CPU RAM (not a full GPT-Neo copy)."""
    return {name: p.detach().cpu().clone()
            for name, p in model.scoring_model.named_parameters() if p.requires_grad}


def _restore_trainable_adapter(model, snapshot):
    if not snapshot:
        raise RuntimeError("Cannot restore best adapter: empty checkpoint")
    with torch.no_grad():
        for name, p in model.scoring_model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name].to(device=p.device, dtype=p.dtype))


def _earlystop_decision(auroc, state, min_delta, patience):
    """Update record-best checkpoint and min_delta-based patience separately."""
    new_record = auroc > state["best_auroc"]
    if new_record:
        state["best_auroc"] = auroc
    significant = auroc >= state["significant_reference"] + min_delta
    if significant:
        state["significant_reference"] = auroc
        state["bad_checks"] = 0
    else:
        state["bad_checks"] += 1
    return new_record, significant, state["bad_checks"] >= patience


def train(model, train_dataset, validation_pairs, split_information):
    from .evaluation import _calculate_validation_auroc
    from .evaluation import _validation_lrp_scores
    """Joint ExDis optimization with real validation AUROC early stopping."""
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if not loader:
        raise ValueError("Empty training dataloader")
    updates_per_epoch = math.ceil(len(loader) / GRAD_ACCUM_STEPS)
    scheduler = CosineAnnealingLR(
        model.scoring_optimizer,
        T_max=max(1, updates_per_epoch * EPOCHS),
        eta_min=0,
    )

    print("=" * 72)
    print("Low-VRAM ExDis training + validation AUROC early stopping")
    print(f"train/validation pairs : {len(train_dataset)}/{len(validation_pairs)}")
    print(f"epochs (max)          : {EPOCHS}")
    print(f"validation/epoch      : {VALIDATIONS_PER_EPOCH}")
    print(f"patience/min_delta    : {EARLY_STOP_PATIENCE}/{EARLY_STOP_MIN_DELTA}")
    print(f"validation source     : {VALIDATION_DATA_PATH} (entire file used)")
    print("DeltaLRP formula      : LRP_optimized - LRP_original")
    print(f"beta/gamma            : {PREFERENCE_BETA}/{PREFERENCE_MARGIN_GAMMA}")
    print(f"batch/accumulation    : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
    print(f"CPU T5 offload        : {AGGRESSIVE_T5_CPU_OFFLOAD}")
    print("=" * 72)

    # Original model scores are invariant during LoRA fine-tuning.
    reference_scores = _validation_lrp_scores(model, validation_pairs, use_reference=True)
    model.set_training_modes()
    earlystop = {"best_auroc": float("-inf"),
                 "significant_reference": float("-inf"), "bad_checks": 0}
    best_adapter = None
    best_step = None
    best_epoch_fraction = None
    global_update = 0
    val_history = []
    stop_reason = "max_epochs"

    for epoch in range(EPOCHS):
        model.sync_old_policy()
        model.set_training_modes()
        model.scoring_optimizer.zero_grad(set_to_none=True)
        # Validate after completed optimizer updates, at ~half and full epoch.
        check_updates = {math.ceil((i + 1) * updates_per_epoch / VALIDATIONS_PER_EPOCH)
                         for i in range(VALIDATIONS_PER_EPOCH)}
        epoch_update = 0
        scoring_losses, ppo_losses, rewards = [], [], []
        ratios, xp_margins, xm_margins = [], [], []
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress):
            human_texts = list(batch[0])
            rewritten_texts = list(batch[1])
            perturbed_texts, masked_inputs, actions = model.generate_perturbations(rewritten_texts)
            reward, _ = model.attack_reward(human_texts, perturbed_texts)
            ppo_stats = model.update_paraphraser(masked_inputs, actions, reward)
            score_out = model.scoring_preference_loss(human_texts, rewritten_texts, perturbed_texts)
            (score_out["loss"] / GRAD_ACCUM_STEPS).backward()
            do_update = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or batch_idx == len(loader) - 1
            if do_update:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.scoring_model.parameters() if p.requires_grad], 1.0)
                model.scoring_optimizer.step()
                model.scoring_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1
                epoch_update += 1

            score_loss_value = float(score_out["loss"].detach().item())
            scoring_losses.append(score_loss_value)
            ppo_losses.append(ppo_stats["total_loss"])
            rewards.append(ppo_stats["reward"])
            ratios.append(ppo_stats["ratio"])
            xp_margins.extend(score_out["perturbed_margin"].cpu().tolist())
            xm_margins.extend(score_out["rewritten_margin"].cpu().tolist())
            progress.set_postfix(score=f"{score_loss_value:.4f}",
                                 ppo=f"{ppo_stats['total_loss']:.4f}",
                                 reward=f"{ppo_stats['reward']:.4f}")

            if do_update and epoch_update in check_updates:
                auroc = _calculate_validation_auroc(model, validation_pairs, reference_scores)
                record_best, significant, must_stop = _earlystop_decision(
                    auroc, earlystop, EARLY_STOP_MIN_DELTA, EARLY_STOP_PATIENCE)
                if record_best:
                    best_adapter = _snapshot_trainable_adapter(model)
                    best_step = global_update
                    best_epoch_fraction = epoch + epoch_update / updates_per_epoch
                val_history.append({"epoch": epoch + 1,
                                    "fractional_epoch": epoch + epoch_update / updates_per_epoch,
                                    "optimizer_step": global_update,
                                    "validation_auroc": auroc,
                                    "record_best_checkpoint": record_best,
                                    "significant_improvement": significant,
                                    "nonimproving_checks": earlystop["bad_checks"]})
                print(f"[Validation] epoch={epoch+1} update={epoch_update}/{updates_per_epoch} "
                      f"AUROC={auroc:.6f} best={earlystop['best_auroc']:.6f} "
                      f"no_improve={earlystop['bad_checks']}/{EARLY_STOP_PATIENCE}")
                model.set_training_modes()
                if must_stop:
                    stop_reason = "patience_exhausted"
                    break

        if scoring_losses:
            print(f"Epoch {epoch+1} scoring loss : {np.mean(scoring_losses):.6f}")
            print(f"Epoch {epoch+1} PPO loss     : {np.mean(ppo_losses):.6f}")
            print(f"Epoch {epoch+1} reward       : {np.mean(rewards):.6f}")
            print(f"Epoch {epoch+1} PPO ratio    : {np.mean(ratios):.6f}")
            print(f"Epoch {epoch+1} xp-h margin  : {np.mean(xp_margins):.6f}")
            print(f"Epoch {epoch+1} xm-h margin  : {np.mean(xm_margins):.6f}")
        if stop_reason == "patience_exhausted":
            break

    _restore_trainable_adapter(model, best_adapter)
    model.set_eval_modes()
    report = {
        "max_epochs": EPOCHS,
        "patience_checks": EARLY_STOP_PATIENCE,
        "min_delta_auroc": EARLY_STOP_MIN_DELTA,
        "validations_per_epoch": VALIDATIONS_PER_EPOCH,
        "best_validation_auroc": earlystop["best_auroc"],
        "best_optimizer_step": best_step,
        "best_fractional_epoch": best_epoch_fraction,
        "stop_reason": stop_reason,
        "total_optimizer_steps": global_update,
        "lrp_score_direction": DELTA_LRP_DIRECTION,
        "model_selected": "validation_best_scoring_LoRA_adapter",
        "training": split_information,
        "validation_history": val_history,
    }
    return model, report



"""Probability and LRP scoring functions used by ExDis."""

import torch
import torch.nn.functional as F

from .config import (
    LOGIT_CHUNK_SIZE,
    PREFERENCE_BETA,
    PREFERENCE_MARGIN_GAMMA,
    SCORING_MAX_LENGTH,
)


def sequence_mean_logprob(model, tokenizer, texts, device, require_grad=True):
    """Compute length-normalized mean token log-probability."""
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SCORING_MAX_LENGTH,
        return_attention_mask=True,
        return_token_type_ids=False,
    ).to(device)

    if encoded.input_ids.shape[1] < 2:
        return torch.zeros(
            len(texts),
            dtype=torch.float32,
            device=device,
        )

    labels = encoded.input_ids[:, 1:]
    valid = encoded.attention_mask[:, 1:].bool()

    def run_forward():
        output = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits[:, :-1, :]
        batch_size = logits.shape[0]

        token_sum = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=device,
        )
        token_count = valid.sum(-1).clamp_min(1).float()
        sequence_length = logits.shape[1]

        for start in range(0, sequence_length, LOGIT_CHUNK_SIZE):
            end = min(start + LOGIT_CHUNK_SIZE, sequence_length)

            chunk = logits[:, start:end, :].float()
            chunk_labels = labels[:, start:end]
            chunk_valid = valid[:, start:end]

            observed = chunk.gather(
                -1,
                chunk_labels.unsqueeze(-1),
            ).squeeze(-1)
            log_partition = torch.logsumexp(chunk, dim=-1)
            token_logprob = observed - log_partition

            token_sum = token_sum + (
                token_logprob * chunk_valid
            ).sum(-1)

            del chunk, observed, log_partition, token_logprob

        del output, logits
        return token_sum / token_count

    if require_grad:
        return run_forward()

    with torch.no_grad():
        return run_forward()


def reference_free_preference_loss(machine_logp, human_logp):
    """Reference-free likelihood preference objective used by ExDis."""
    margin = machine_logp - human_logp
    loss = -F.logsigmoid(
        PREFERENCE_BETA * (margin - PREFERENCE_MARGIN_GAMMA)
    ).mean()
    return loss, margin


@torch.no_grad()
def calculate_lrp(model, tokenizer, text, device):
    """Compute LRPscore = mean(log likelihood) * mean(log rank)."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=SCORING_MAX_LENGTH,
        return_attention_mask=True,
        return_token_type_ids=False,
    ).to(device)

    if encoded.input_ids.shape[1] < 2:
        return float("nan")

    labels = encoded.input_ids[:, 1:]
    valid = encoded.attention_mask[:, 1:].bool()

    output = model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits[:, :-1, :]

    logprob_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    logrank_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    count = 0
    sequence_length = logits.shape[1]

    for start in range(0, sequence_length, LOGIT_CHUNK_SIZE):
        end = min(start + LOGIT_CHUNK_SIZE, sequence_length)

        chunk = logits[:, start:end, :].float()
        chunk_labels = labels[:, start:end]
        chunk_valid = valid[:, start:end]

        observed_logits = chunk.gather(
            -1,
            chunk_labels.unsqueeze(-1),
        ).squeeze(-1)

        observed_logprob = observed_logits - torch.logsumexp(
            chunk,
            dim=-1,
        )

        rank = (
            (chunk > observed_logits.unsqueeze(-1))
            .sum(-1)
            .float()
            + 1.0
        )
        log_rank = torch.log(rank)

        selected_logprob = observed_logprob[chunk_valid]
        selected_logrank = log_rank[chunk_valid]

        if selected_logprob.numel() > 0:
            logprob_sum += selected_logprob.sum()
            logrank_sum += selected_logrank.sum()
            count += int(selected_logprob.numel())

        del (
            chunk,
            observed_logits,
            observed_logprob,
            rank,
            log_rank,
            selected_logprob,
            selected_logrank,
        )

    del output, logits, encoded

    if count == 0:
        return float("nan")

    mean_logprob = logprob_sum / count
    mean_logrank = logrank_sum / count
    return float((mean_logprob * mean_logrank).item())

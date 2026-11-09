"""ExDis scoring module."""
from .deps import *

def sequence_mean_logprob(model, tokenizer, texts, device, require_grad=True):
    """
    Length-normalized mean token log-probability.

    Memory-safe implementation:
    the model still produces its logits normally, but conversion to FP32 and
    logsumexp are performed in small time-axis chunks instead of materializing
    a full FP32 log-softmax tensor for all tokens and vocabulary entries.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SCORING_MAX_LENGTH,
        return_attention_mask=True,
        return_token_type_ids=False,
    ).to(device)

    if enc.input_ids.shape[1] < 2:
        return torch.zeros(
            len(texts),
            dtype=torch.float32,
            device=device
        )

    labels = enc.input_ids[:, 1:]
    valid = enc.attention_mask[:, 1:].bool()

    def _run():
        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = out.logits[:, :-1, :]
        batch_size = logits.shape[0]

        token_sum = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=device
        )
        token_count = valid.sum(-1).clamp_min(1).float()

        seq_len = logits.shape[1]

        for start in range(0, seq_len, LOGIT_CHUNK_SIZE):
            end = min(
                start + LOGIT_CHUNK_SIZE,
                seq_len
            )

            chunk = logits[:, start:end, :].float()
            chunk_labels = labels[:, start:end]
            chunk_valid = valid[:, start:end]

            observed = chunk.gather(
                -1,
                chunk_labels.unsqueeze(-1)
            ).squeeze(-1)

            log_z = torch.logsumexp(
                chunk,
                dim=-1
            )

            token_lp = observed - log_z

            token_sum = token_sum + (
                token_lp * chunk_valid
            ).sum(-1)

            del chunk, observed, log_z, token_lp

        del out, logits

        return token_sum / token_count

    if require_grad:
        return _run()

    with torch.no_grad():
        return _run()


@torch.no_grad()
def calculate_lrp(model, tokenizer, text, device):
    """
    LRPscore = mean(log likelihood) * mean(log rank)

    Memory-safe implementation: FP32 conversion, logsumexp and rank counting
    are performed in short token chunks. This is slower but greatly reduces
    temporary GPU memory.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=SCORING_MAX_LENGTH,
        return_attention_mask=True,
        return_token_type_ids=False,
    ).to(device)

    if enc.input_ids.shape[1] < 2:
        return float("nan")

    labels = enc.input_ids[:, 1:]
    valid = enc.attention_mask[:, 1:].bool()

    out = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = out.logits[:, :-1, :]

    lp_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=device
    )
    rank_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=device
    )
    count = 0

    seq_len = logits.shape[1]

    for start in range(0, seq_len, LOGIT_CHUNK_SIZE):
        end = min(
            start + LOGIT_CHUNK_SIZE,
            seq_len
        )

        chunk = logits[:, start:end, :].float()
        chunk_labels = labels[:, start:end]
        chunk_valid = valid[:, start:end]

        observed_logits = chunk.gather(
            -1,
            chunk_labels.unsqueeze(-1)
        ).squeeze(-1)

        observed_lp = (
            observed_logits
            - torch.logsumexp(chunk, dim=-1)
        )

        rank = (
            (
                chunk
                > observed_logits.unsqueeze(-1)
            )
            .sum(-1)
            .float()
            + 1.0
        )

        log_rank = torch.log(rank)

        selected_lp = observed_lp[
            chunk_valid
        ]
        selected_rank = log_rank[
            chunk_valid
        ]

        if selected_lp.numel() > 0:
            lp_sum += selected_lp.sum()
            rank_sum += selected_rank.sum()
            count += int(selected_lp.numel())

        del (
            chunk,
            observed_logits,
            observed_lp,
            rank,
            log_rank,
            selected_lp,
            selected_rank,
        )

    del out, logits, enc

    if count == 0:
        return float("nan")

    mean_lp = lp_sum / count
    mean_rank = rank_sum / count

    return float(
        (mean_lp * mean_rank).item()
    )


def delta_lrp_score(original_lrp, optimized_lrp):
    """Fixed ExDis sign: optimized-model LRP minus original-model LRP.

    This is used by BOTH early-stopping validation and final test AUROC.
    """
    return optimized_lrp - original_lrp



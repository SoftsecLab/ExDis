"""T5 sentinel-span corruption and reconstruction utilities."""

import random

from .config import T5_MAX_LENGTH


def group_consecutive_positions(positions):
    """Convert token positions into half-open consecutive spans."""
    if not positions:
        return []

    positions = sorted(set(int(position) for position in positions))
    spans = []
    start = positions[0]
    previous = positions[0]

    for position in positions[1:]:
        if position == previous + 1:
            previous = position
        else:
            spans.append((start, previous + 1))
            start = previous = position

    spans.append((start, previous + 1))
    return spans


def build_t5_sentinel_input(tokenizer, text, ratio):
    """Mask a random token subset using T5 sentinel tokens."""
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=T5_MAX_LENGTH - 2,
    )

    if len(ids) < 4:
        return None

    num_mask = max(1, int(round(len(ids) * ratio)))
    num_mask = min(num_mask, len(ids) - 1)
    positions = random.sample(range(len(ids)), num_mask)
    spans = group_consecutive_positions(positions)[:100]
    starts = {start: (index, end) for index, (start, end) in enumerate(spans)}

    masked_ids = []
    position = 0

    while position < len(ids):
        if position in starts:
            span_index, end = starts[position]
            sentinel_id = tokenizer.convert_tokens_to_ids(
                f"<extra_id_{span_index}>"
            )
            if sentinel_id is None or sentinel_id == tokenizer.unk_token_id:
                raise RuntimeError(
                    "T5 tokenizer lacks <extra_id_n> sentinel tokens."
                )
            masked_ids.append(sentinel_id)
            position = end
        else:
            masked_ids.append(ids[position])
            position += 1

    masked_text = tokenizer.decode(
        masked_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "original_ids": ids,
        "spans": spans,
        "masked_text": masked_text,
    }


def strip_generated_action_ids(tokenizer, generated_ids):
    """Remove leading padding from a generated T5 action sequence."""
    ids = generated_ids.tolist()

    while (
        ids
        and tokenizer.pad_token_id is not None
        and ids[0] == tokenizer.pad_token_id
    ):
        ids = ids[1:]

    if not ids:
        ids = [
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else 0
        ]

    return ids


def parse_t5_generated_fills(tokenizer, action_ids, num_spans):
    """Map generated tokens back to their corresponding sentinel spans."""
    sentinel_to_index = {
        tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"): index
        for index in range(num_spans)
    }
    fills = {index: [] for index in range(num_spans)}
    current = None

    for token_id in action_ids:
        if token_id in sentinel_to_index:
            current = sentinel_to_index[token_id]
            continue
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        if tokenizer.pad_token_id is not None and token_id == tokenizer.pad_token_id:
            continue
        if current is not None:
            fills[current].append(token_id)

    return fills


def reconstruct_perturbed_text(tokenizer, mask_info, action_ids):
    """Reconstruct perturbed text from T5-generated span fills."""
    original_ids = mask_info["original_ids"]
    spans = mask_info["spans"]
    fills = parse_t5_generated_fills(tokenizer, action_ids, len(spans))
    starts = {start: (index, end) for index, (start, end) in enumerate(spans)}

    output = []
    position = 0

    while position < len(original_ids):
        if position in starts:
            span_index, end = starts[position]
            replacement = fills.get(span_index, [])
            if not replacement:
                replacement = original_ids[position:end]
            output.extend(replacement)
            position = end
        else:
            output.append(original_ids[position])
            position += 1

    return tokenizer.decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

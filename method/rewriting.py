"""ExDis rewriting module."""
from .deps import *

def group_consecutive_positions(positions):
    if not positions:
        return []
    positions = sorted(set(int(x) for x in positions))
    spans = []
    s = positions[0]
    p = positions[0]
    for x in positions[1:]:
        if x == p + 1:
            p = x
        else:
            spans.append((s, p + 1))
            s = p = x
    spans.append((s, p + 1))
    return spans


def build_t5_sentinel_input(tokenizer, text, ratio):
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
    starts = {s: (j, e) for j, (s, e) in enumerate(spans)}
    masked_ids = []
    i = 0
    while i < len(ids):
        if i in starts:
            j, e = starts[i]
            sid = tokenizer.convert_tokens_to_ids(f"<extra_id_{j}>")
            if sid is None or sid == tokenizer.unk_token_id:
                raise RuntimeError("T5 tokenizer lacks <extra_id_n> sentinel tokens.")
            masked_ids.append(sid)
            i = e
        else:
            masked_ids.append(ids[i])
            i += 1
    masked_text = tokenizer.decode(
        masked_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {"original_ids": ids, "spans": spans, "masked_text": masked_text}


def strip_generated_action_ids(tokenizer, generated_ids):
    ids = generated_ids.tolist()
    while ids and tokenizer.pad_token_id is not None and ids[0] == tokenizer.pad_token_id:
        ids = ids[1:]
    if not ids:
        ids = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0]
    return ids


def parse_t5_generated_fills(tokenizer, action_ids, num_spans):
    sentinel_to_index = {
        tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>"): i
        for i in range(num_spans)
    }
    fills = {i: [] for i in range(num_spans)}
    current = None
    for tid in action_ids:
        if tid in sentinel_to_index:
            current = sentinel_to_index[tid]
            continue
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break
        if tokenizer.pad_token_id is not None and tid == tokenizer.pad_token_id:
            continue
        if current is not None:
            fills[current].append(tid)
    return fills


def reconstruct_perturbed_text(tokenizer, mask_info, action_ids):
    original_ids = mask_info["original_ids"]
    spans = mask_info["spans"]
    fills = parse_t5_generated_fills(tokenizer, action_ids, len(spans))
    starts = {s: (j, e) for j, (s, e) in enumerate(spans)}
    output = []
    i = 0
    while i < len(original_ids):
        if i in starts:
            j, e = starts[i]
            replacement = fills.get(j, [])
            if not replacement:
                replacement = original_ids[i:e]
            output.extend(replacement)
            i = e
        else:
            output.append(original_ids[i])
            i += 1
    return tokenizer.decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()



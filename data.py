"""Dataset utilities for ExDis."""

import json
from pathlib import Path

from torch.utils.data import Dataset


class AlignedRewriteDataset(Dataset):
    """Load aligned human/rewrite pairs from a JSON file."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            obj = json.load(file)

        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON must be a dictionary.")
        if "original" not in obj or "rewritten" not in obj:
            raise KeyError("JSON must contain 'original' and 'rewritten'.")

        original = obj["original"]
        rewritten = obj["rewritten"]

        if not isinstance(original, list) or not isinstance(rewritten, list):
            raise ValueError("'original' and 'rewritten' must be lists.")
        if len(original) != len(rewritten):
            raise ValueError(
                f"Aligned pairs required: original={len(original)}, "
                f"rewritten={len(rewritten)}"
            )

        self.pairs = [
            (human.strip(), machine.strip())
            for human, machine in zip(original, rewritten)
            if isinstance(human, str)
            and isinstance(machine, str)
            and human.strip()
            and machine.strip()
        ]

        if not self.pairs:
            raise ValueError("No valid aligned pairs.")

        print(f"{path.name}: aligned pairs={len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

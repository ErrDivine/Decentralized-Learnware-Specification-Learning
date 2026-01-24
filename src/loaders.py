"""Dataset loading helpers."""

from __future__ import annotations

from datasets import load_from_disk, DatasetDict


def load_split_from_disk(path: str, split: str):
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        return ds[split]
    return ds

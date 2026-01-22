"""Data loading utilities for the Head module.

The Head expects two heterogeneous vectors: task (t) and variables (v).
This loader keeps the two streams separate per the design in docs/theories.md
and returns batches as dictionaries compatible with Head.fit/training_step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_from_disk
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "datasets library is required for data loading. Install with `pip install datasets`."
    ) from exc


TaskEncoder = Callable[[Dict[str, Any]], Tensor]
VariablesEncoder = Callable[[Dict[str, Any]], Tensor]
TargetEncoder = Callable[[Dict[str, Any]], Tensor]


@dataclass
class LoaderConfig:
    dataset_path: Union[str, Path]
    split: str = "train"
    dataset_name: Optional[str] = None  # e.g., "gsm8k" or "hendrycks_math"
    batch_size: int = 8
    shuffle: Optional[bool] = None
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    device: Optional[Union[torch.device, str]] = None
    constant_target: Optional[float] = None  # fallback when labels are not stored
    extra_fields: Optional[Sequence[str]] = None  # e.g., actions, old_log_probs, advantages


def _to_float_tensor(x: Any, name: str) -> Tensor:
    if isinstance(x, Tensor):
        return x.float()
    try:
        return torch.as_tensor(x, dtype=torch.float32)
    except Exception as exc:
        raise TypeError(f"Could not convert {name} to tensor; got type {type(x)}") from exc


def _text_stats_vector(text: str) -> Tensor:
    """Lightweight numeric stats to turn text into a fixed-size vector."""
    if not isinstance(text, str):
        raise TypeError(f"Expected string for text stats, got {type(text)}")
    words = text.split()
    chars = len(text)
    digits = sum(ch.isdigit() for ch in text)
    symbols = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    avg_word_len = chars / max(1, len(words))
    return torch.tensor([chars, len(words), digits, symbols, avg_word_len], dtype=torch.float32)


def _hash_bucket(value: Any, buckets: int) -> float:
    """Stable bucketized hash scaled to [0,1] for categorical metadata."""
    if buckets < 1:
        raise ValueError("buckets must be >= 1")
    hashed = hash(value) % buckets
    return float(hashed) / max(1, buckets - 1)


def default_encoders_for_dataset(dataset_name: str) -> Tuple[TaskEncoder, VariablesEncoder]:
    """Provide simple encoders aligned with available splits on phoebe branch."""
    name = dataset_name.lower()
    if name == "gsm8k":
        return (
            lambda ex: _text_stats_vector(ex["question"]),
            lambda ex: _text_stats_vector(ex["answer"]),
        )
    if name == "hendrycks_math":
        def variables(ex: Dict[str, Any]) -> Tensor:
            solution_stats = _text_stats_vector(ex["solution"])
            meta = torch.tensor(
                [
                    _hash_bucket(ex.get("level", ""), buckets=32),
                    _hash_bucket(ex.get("type", ""), buckets=64),
                ],
                dtype=torch.float32,
            )
            return torch.cat([solution_stats, meta])

        return (
            lambda ex: _text_stats_vector(ex["problem"]),
            variables,
        )
    raise ValueError(f"No default encoders for dataset '{dataset_name}'")


class HeadDataset(Dataset):
    """Wrap a Hugging Face dataset to emit task/variables/targets tensors."""

    def __init__(
        self,
        dataset: HFDataset,
        task_encoder: TaskEncoder,
        variables_encoder: VariablesEncoder,
        target_encoder: Optional[TargetEncoder] = None,
        constant_target: Optional[float] = None,
        extra_fields: Optional[Sequence[str]] = None,
    ) -> None:
        if target_encoder is None and constant_target is None:
            raise ValueError("Provide a target_encoder or constant_target for labels.")
        self.dataset = dataset
        self.task_encoder = task_encoder
        self.variables_encoder = variables_encoder
        self.target_encoder = target_encoder
        self.constant_target = constant_target
        self.extra_fields = list(extra_fields) if extra_fields else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.dataset[idx]
        task = _to_float_tensor(self.task_encoder(sample), "task")
        variables = _to_float_tensor(self.variables_encoder(sample), "variables")
        if self.target_encoder is not None:
            targets = _to_float_tensor(self.target_encoder(sample), "targets")
        else:
            targets = torch.tensor(self.constant_target, dtype=torch.float32)
        data = {"task": task, "variables": variables, "targets": targets}
        for key in self.extra_fields:
            if key not in sample:
                raise KeyError(f"Extra field '{key}' not found in sample.")
            data[key] = _to_float_tensor(sample[key], key)
        return data


class HeadCollator:
    """Collate samples into batched tensors with optional device placement."""

    def __init__(
        self, device: Optional[Union[torch.device, str]] = None, extra_fields=None
    ) -> None:
        self.device = device
        self.extra_fields = list(extra_fields) if extra_fields else []

    def __call__(self, batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        tasks = torch.stack([b["task"] for b in batch], dim=0)
        variables = torch.stack([b["variables"] for b in batch], dim=0)
        targets = torch.stack([b["targets"] for b in batch], dim=0)
        extras = {}
        for key in self.extra_fields:
            extras[key] = torch.stack([b[key] for b in batch], dim=0)
        if self.device is not None:
            tasks = tasks.to(self.device)
            variables = variables.to(self.device)
            targets = targets.to(self.device)
            extras = {k: v.to(self.device) for k, v in extras.items()}
        return {"task": tasks, "variables": variables, "targets": targets, **extras}


def load_split_from_disk(path: Union[str, Path], split: str) -> HFDataset:
    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
        return ds[split]
    return ds


def build_head_dataloader(
    config: LoaderConfig,
    task_encoder: Optional[TaskEncoder] = None,
    variables_encoder: Optional[VariablesEncoder] = None,
    target_encoder: Optional[TargetEncoder] = None,
) -> DataLoader:
    """Construct a DataLoader that yields dict batches compatible with Head.fit."""
    dataset = load_split_from_disk(config.dataset_path, config.split)

    if task_encoder is None or variables_encoder is None:
        if config.dataset_name is None:
            raise ValueError("dataset_name is required to use default encoders.")
        task_encoder, variables_encoder = default_encoders_for_dataset(config.dataset_name)

    if target_encoder is None and config.constant_target is None:
        raise ValueError("Specify target_encoder or set constant_target for labels.")

    wrapped = HeadDataset(
        dataset=dataset,
        task_encoder=task_encoder,
        variables_encoder=variables_encoder,
        target_encoder=target_encoder,
        constant_target=config.constant_target,
        extra_fields=config.extra_fields,
    )

    collate = HeadCollator(device=config.device, extra_fields=config.extra_fields)
    shuffle = config.shuffle
    if shuffle is None:
        shuffle = config.split == "train"

    return DataLoader(
        wrapped,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate,
    )

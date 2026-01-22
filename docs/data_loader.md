# Data Loader

Utilities for building PyTorch DataLoaders that emit batches compatible with `Head` training (supervised or PPO). By default it wraps Hugging Face datasets saved with `datasets.save_to_disk`.

## LoaderConfig
```python
LoaderConfig(
    dataset_path,         # path to HF dataset (Dataset or DatasetDict on disk)
    split="train",
    dataset_name=None,    # "gsm8k" or "hendrycks_math" to use default encoders
    batch_size=8,
    shuffle=None,         # defaults to True for train, False otherwise
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    device=None,          # move batches to device in collator
    constant_target=None, # fallback label for supervised mode
    extra_fields=None     # iterable of extra keys to carry (e.g., actions, old_log_probs)
)
```

## Encoders
- `default_encoders_for_dataset(name)` provides simple numeric encodings:
  - `gsm8k`: task = text stats of `question`; variables = text stats of `answer`
  - `hendrycks_math`: task = text stats of `problem`; variables = text stats of `solution` plus hashed `level`/`type`
- You can pass custom `task_encoder`, `variables_encoder`, and `target_encoder` to `build_head_dataloader`. For PPO, targets are usually unused; you can set `constant_target=0.0` and use `extra_fields` to carry PPO signals.

Text stats encoder produces `[chars, words, digits, symbols, avg_word_len]` (float tensor).

## Datasets and batches
- `HeadDataset` wraps an HF Dataset; `__getitem__` returns a dict with:
  - `task` (float tensor), `variables` (float tensor), `targets` (float tensor)
  - any `extra_fields` requested (stacked and moved to device by the collator)
- `HeadCollator` stacks each field into batch tensors and moves them to `device` if provided.

## Building a loader
```python
from src.modules.data_loader import LoaderConfig, build_head_dataloader

cfg = LoaderConfig(
    dataset_path="datasets/gsm8k_arrow",
    dataset_name="gsm8k",
    split="train",
    batch_size=16,
    constant_target=0.0,          # if not using supervised labels
    extra_fields=["actions", "old_log_probs", "advantages", "value_targets"],
    device="cuda"
)
dl = build_head_dataloader(cfg)
```

For supervised pretraining, provide a `target_encoder` or set `constant_target`. For PPO, store rollout buffers back to disk (with the needed keys) or implement a small in-memory Dataset and keep using the same collator pattern.

## Expected fields per training mode
- **Supervised**: `task`, `variables`, `targets`
- **PPO** (`Head.fit(..., supervised=False)` or `ppo_loss`): `task`, `variables`, `actions`, `old_log_probs`, `advantages`; optional `value_targets` if training the critic.

## Notes
- `extra_fields` must exist in the dataset samples; missing keys raise errors to avoid silent bugs.
- Collator keeps shapes; ensure your encoders output consistent dimensions aligned with `HeadConfig.task_dim` and `var_dim`.
- If you need a replay-buffer dataset instead of HF disk data, wrap your buffer in a `torch.utils.data.Dataset` that returns the same dict schema and reuse `HeadCollator`.

"""Evaluate saved Heads on a dataset split.

Loads per-agent checkpoints, runs deterministic votes (Beta mean) over a dataset,
and reports average votes and winner rates. Useful to compare multi-agent heads
against a single-head baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modules.data_loader import (  # type: ignore  # noqa: E402
    LoaderConfig,
    build_head_dataloader,
)
from src.modules.head import Head, HeadConfig  # type: ignore  # noqa: E402


def load_heads(
    ckpt_dir: Path, n_agents: int, head_cfg: HeadConfig, device: torch.device
) -> List[Head]:
    heads: List[Head] = []
    for idx in range(n_agents):
        # pick the latest checkpoint for this agent
        candidates = sorted(ckpt_dir.glob(f"head_agent{idx}_epoch*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for agent {idx} in {ckpt_dir}")
        ckpt = candidates[-1]
        head = Head(head_cfg).to(device)
        head.load_state_dict(torch.load(ckpt, map_location=device))
        head.eval()
        heads.append(head)
    return heads


def evaluate(
    heads: List[Head],
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    vote_sums = torch.zeros(len(heads), device=device)
    counts = 0
    winner_counts = torch.zeros(len(heads), device=device)

    with torch.no_grad():
        for batch in dataloader:
            t = batch["task"].to(device)
            v = batch["variables"].to(device)
            batch_size = t.shape[0]
            votes = torch.stack([head.predict_vote(t, v) for head in heads], dim=1)  # [B, n_agents]
            vote_sums += votes.sum(dim=0)
            winners = votes.argmax(dim=1)
            winner_counts += torch.bincount(winners, minlength=len(heads))
            counts += batch_size

    avg_votes = (vote_sums / counts).tolist()
    win_rates = (winner_counts / counts).tolist()
    for idx, (v, w) in enumerate(zip(avg_votes, win_rates)):
        print(f"Agent {idx}: avg vote={v:.4f}, win_rate={w:.4f}")
    best = int(torch.argmax(torch.tensor(win_rates)))
    print(f"Top agent by win_rate: {best}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved Heads on a dataset split.")
    ap.add_argument("--ckpt-dir", required=True, help="Directory with head_agent{idx}_epoch*.pt files.")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--dataset-name", required=True, choices=["gsm8k", "hendrycks_math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--ff-dim", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Build a dataloader using the same encoders; targets are unused
    cfg = LoaderConfig(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        constant_target=0.0,
        extra_fields=[],
        device=device,
    )
    dl = build_head_dataloader(cfg)

    # Infer dims from one batch
    batch = next(iter(dl))
    t_dim = batch["task"].shape[-1]
    v_dim = batch["variables"].shape[-1]

    # Discover number of agents from checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    agents = sorted({p.stem.split("_")[1].replace("agent", "") for p in ckpt_dir.glob("head_agent*_epoch*.pt")})
    n_agents = len(agents)
    if n_agents == 0:
        raise FileNotFoundError(f"No head checkpoints in {ckpt_dir}")

    head_cfg = HeadConfig(task_dim=t_dim, var_dim=v_dim, hidden_dim=args.hidden_dim, ff_dim=args.ff_dim, use_critic=False)
    heads = load_heads(ckpt_dir, n_agents, head_cfg, device)
    evaluate(heads, dl, device)


if __name__ == "__main__":
    main()

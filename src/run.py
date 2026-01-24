"""Train Heads with PPO on real env: LLM agents + judger, no synthetic paths."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

# Ensure project root is on path when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modules.envs import RealEnv, TextEncoder
from src.agents.reason_agent import ReasonAgent
from src.agents.compute_agent import ComputeAgent
from src.agents.model_agent import ModelAgent
from src.agents.verify_agent import VerifyAgent
from src.modules.head import Head, HeadConfig


class PPODataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Tensor]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self.records[idx]


def build_dataloader(
    records: Sequence[Dict[str, Tensor]], batch_size: int, device: torch.device
) -> DataLoader:
    def collate(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        keys = batch[0].keys()
        out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
        out = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in out.items()}
        return out

    return DataLoader(
        PPODataset(records), batch_size=batch_size, shuffle=True, collate_fn=collate
    )


@torch.no_grad()
def collect_rollouts(
    env: RealEnv,
    heads: Sequence[Head],
    steps: int,
    device: torch.device,
    stop_threshold: float,
) -> Dict[int, List[Dict[str, Tensor]]]:
    buffers: Dict[int, List[Dict[str, Tensor]]] = {i: [] for i in range(len(heads))}
    state_t, state_v = env.reset()
    state_t = state_t.to(device)
    state_v = state_v.to(device)

    for _ in range(steps):
        t_cur = state_t.detach()
        v_cur = state_v.detach()

        votes, logps, values = [], [], []
        for head in heads:
            vote, logp, _, _ = head.sample_vote(t_cur, v_cur, return_aux=True)
            votes.append(vote.squeeze())
            logps.append(logp.squeeze())
            val = (
                head.critic_value(t_cur, v_cur)
                if head.config.use_critic
                else torch.tensor(0.0, device=device)
            )
            values.append(val.squeeze())

        votes_tensor = torch.stack(votes)
        if votes_tensor.max().item() < stop_threshold:
            state_t, state_v = env.reset()
            state_t = state_t.to(device)
            state_v = state_v.to(device)
            continue

        chosen = int(torch.argmax(votes_tensor).item())
        vote_chosen = votes[chosen].detach()
        logp_chosen = logps[chosen].detach()
        value_pred = values[chosen].detach()

        reward, (state_t, state_v), _, done = env.step(chosen, float(vote_chosen))
        state_t = state_t.to(device)
        state_v = state_v.to(device)

        advantage = torch.tensor(reward, device=device) - value_pred

        record = {
            "task": t_cur.float(),
            "variables": v_cur.float(),
            "actions": vote_chosen.float(),
            "old_log_probs": logp_chosen.float(),
            "advantages": advantage.float(),
            "value_targets": torch.tensor(reward, dtype=torch.float32, device=device),
        }
        buffers[chosen].append(record)

        if done:
            state_t, state_v = env.reset()
            state_t = state_t.to(device)
            state_v = state_v.to(device)

    return buffers


def ppo_update(
    head: Head,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> List[float]:
    head.train()
    losses: List[float] = []
    for batch in dataloader:
        loss, _ = head.ppo_loss(
            task=batch["task"],
            variables=batch["variables"],
            actions=batch["actions"],
            old_log_probs=batch["old_log_probs"],
            advantages=batch["advantages"],
            clip_eps=clip_eps,
            value_targets=batch.get("value_targets"),
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def make_real_env(
    dataset_path: str,
    split: str,
    encoder_model_path: str,
    judger_model: str,
    judger_region: str,
    max_steps: int,
    device: torch.device,
    agent_max_new_tokens: int,
) -> RealEnv:
    ds = load_from_disk(dataset_path)[split]
    encoder = TextEncoder(encoder_model_path, device=device)
    agents = [ComputeAgent(), ReasonAgent(), ModelAgent(), VerifyAgent()]
    return RealEnv(
        dataset=ds,
        encoder=encoder,
        agents=agents,
        judger_model=judger_model,
        judger_region=judger_region,
        max_steps=max_steps,
        device=device,
        agent_max_new_tokens=agent_max_new_tokens,
    )


def save_heads(heads: Sequence[Head], out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, head in enumerate(heads):
        ckpt_path = out_dir / f"head_agent{idx}_epoch{epoch}.pt"
        torch.save(head.state_dict(), ckpt_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Heads with PPO on real env (LLM agents + judger).")
    ap.add_argument("--dataset-path", default="datasets/gsm8k_arrow")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--ppo-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--stop-threshold", type=float, default=0.05, help="Terminate episode early if max vote falls below this.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="runs/heads_ckpt")
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--encoder-model-path", default="model/encoder")
    ap.add_argument("--judger-model", default="qwen-max")
    ap.add_argument("--judger-region", default="beijing")
    ap.add_argument("--agent-max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = make_real_env(
        dataset_path=args.dataset_path,
        split=args.split,
        encoder_model_path=args.encoder_model_path,
        judger_model=args.judger_model,
        judger_region=args.judger_region,
        max_steps=args.max_steps,
        device=device,
        agent_max_new_tokens=args.agent_max_new_tokens,
    )

    sample_t, sample_v = env.encoded_state
    task_dim = sample_t.numel()
    var_dim = sample_v.numel()

    head_cfg = HeadConfig(
        task_dim=task_dim,
        var_dim=var_dim,
        hidden_dim=512,
        ff_dim=1024,
        use_critic=True,
    )
    heads = [Head(head_cfg).to(device) for _ in range(4)]
    optimizers = [Head.build_optimizer(h, lr=args.lr) for h in heads]

    save_dir = Path(args.save_dir)

    for epoch in range(1, args.ppo_epochs + 1):
        buffers = collect_rollouts(env, heads, args.rollout_steps, device, args.stop_threshold)
        total_samples = sum(len(r) for r in buffers.values())
        epoch_losses: List[float] = []
        for agent_idx, records in buffers.items():
            if not records:
                continue
            dl = build_dataloader(records, batch_size=args.batch_size, device=device)
            losses = ppo_update(
                heads[agent_idx],
                optimizers[agent_idx],
                dl,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )
            epoch_losses.extend(losses)
        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch}: mean PPO loss={mean_loss:.4f}, samples={total_samples}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_heads(heads, save_dir, epoch)

    with torch.no_grad():
        t_eval, v_eval = env.encoded_state
        t_eval = t_eval.to(device)
        v_eval = v_eval.to(device)
        votes = []
        for idx, head in enumerate(heads):
            vote = head.predict_vote(t_eval, v_eval)
            votes.append(vote.item())
            print(f"Agent {idx} vote (mean): {vote.item():.4f}")
        winner = int(torch.argmax(torch.tensor(votes)))
        print(f"Winner: agent {winner}")


if __name__ == "__main__":
    main()

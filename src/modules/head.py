"""Head module for decentralized learnware specification learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    task_dim: int
    var_dim: int
    hidden_dim: int = 256
    attention_heads: int = 4
    ff_dim: int = 256
    encoder_depth: int = 2
    dropout: float = 0.1
    use_critic: bool = True


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Head(nn.Module):
    """Head module producing a Beta policy over votes Pr(action=True|t,v)."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.attention_heads < 1:
            raise ValueError("attention_heads must be >= 1")

        self.task_encoder = MLPEncoder(
            config.task_dim,
            config.hidden_dim,
            depth=config.encoder_depth,
            dropout=config.dropout,
        )
        self.var_encoder = MLPEncoder(
            config.var_dim,
            config.hidden_dim,
            depth=config.encoder_depth,
            dropout=config.dropout,
        )

        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.attn_norm = nn.LayerNorm(config.hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, 2),
        )

        self.critic = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.ff_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_dim, 1),
            )
            if config.use_critic
            else None
        )

    def _ensure_3d(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(1)
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def _encode(
        self, task: Tensor, variables: Tensor, return_attention: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        t = self.task_encoder(task)
        v = self.var_encoder(variables)

        t_seq = self._ensure_3d(t)
        v_seq = self._ensure_3d(v)

        attn_out, attn_weights = self.cross_attention(
            t_seq,
            v_seq,
            v_seq,
            need_weights=return_attention,
        )
        attn_out = self.attn_norm(t_seq + self.attn_dropout(attn_out))
        pooled = attn_out.mean(dim=1)
        return pooled, attn_weights if return_attention else None

    def forward(
        self,
        task: Tensor,
        variables: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return (alpha,beta) parameters of Beta policy."""
        pooled, attn = self._encode(task, variables, return_attention)
        alpha_beta = F.softplus(self.policy_head(pooled)) + 1e-4
        if return_attention:
            return alpha_beta, attn
        return alpha_beta

    def distribution(
        self, task: Tensor, variables: Tensor
    ) -> torch.distributions.Beta:
        alpha_beta = self.forward(task, variables)
        alpha, beta = alpha_beta.chunk(2, dim=-1)
        return torch.distributions.Beta(alpha, beta)

    def sample_vote(
        self,
        task: Tensor,
        variables: Tensor,
        deterministic: bool = False,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Sample (or take mean) vote in [0,1] with log prob for PPO updates."""
        dist = self.distribution(task, variables)
        vote = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(vote.clamp(1e-6, 1 - 1e-6))
        if return_aux:
            return vote, log_prob, dist.concentration1, dist.concentration0
        return vote

    def critic_value(self, task: Tensor, variables: Tensor) -> Tensor:
        if self.critic is None:
            raise ValueError("Critic is disabled in config.")
        pooled, _ = self._encode(task, variables, return_attention=False)
        return self.critic(pooled).squeeze(-1)

    def predict_vote(
        self,
        task: Tensor,
        variables: Tensor,
    ) -> Tensor:
        """Return deterministic vote (mean of Beta) in [0,1]."""
        self.eval()
        with torch.no_grad():
            dist = self.distribution(task, variables)
            return dist.mean.squeeze(-1)

    def should_act(
        self,
        task: Tensor,
        variables: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Return a boolean decision tensor based on threshold."""
        vote = self.predict_vote(task, variables)
        return vote >= threshold

    def ppo_loss(
        self,
        task: Tensor,
        variables: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        clip_eps: float = 0.2,
        value_targets: Optional[Tensor] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute PPO loss for Beta policy as described in docs/theories.md."""
        dist = self.distribution(task, variables)
        actions = actions.clamp(1e-6, 1 - 1e-6)
        log_probs = dist.log_prob(actions)
        ratios = (log_probs - old_log_probs).exp()

        advantages = advantages
        clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()

        entropy = dist.entropy().mean()

        value_loss = torch.tensor(0.0, device=policy_loss.device)
        if value_targets is not None:
            if self.critic is None:
                raise ValueError("value_targets provided but critic is disabled.")
            value_preds = self.critic_value(task, variables)
            value_loss = F.mse_loss(value_preds, value_targets)

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        stats = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, stats

    def fit(
        self,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = 1.0,
        supervised: bool = True,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
    ) -> Sequence[float]:
        """Training loop for either supervised (legacy) or PPO batches."""
        self.train()
        losses = []
        for _ in range(epochs):
            for batch in dataloader:
                if device is not None:
                    batch = self._move_batch(batch, device)

                if supervised:
                    loss = self._training_step_supervised(batch)
                else:
                    loss, _ = self._training_step_ppo(
                        batch,
                        clip_eps=clip_eps,
                        value_coef=value_coef,
                        entropy_coef=entropy_coef,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

    def _training_step_supervised(
        self, batch: Union[Dict[str, Tensor], Sequence[Tensor]]
    ) -> Tensor:
        if isinstance(batch, dict):
            task = batch.get("task")
            variables = batch.get("variables")
            targets = batch.get("targets")
        else:
            if len(batch) < 3:
                raise ValueError("batch must provide task, variables, and targets")
            task, variables, targets = batch[:3]

        if targets is None:
            raise ValueError("Supervised training requires targets.")

        dist = self.distribution(task, variables)
        targets = targets.clamp(1e-6, 1 - 1e-6)
        log_probs = dist.log_prob(targets)
        # Maximize log likelihood of targets -> minimize negative log prob
        return -log_probs.mean()

    def _training_step_ppo(
        self,
        batch: Dict[str, Tensor],
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        required = ["task", "variables", "actions", "old_log_probs", "advantages"]
        for key in required:
            if key not in batch:
                raise ValueError(f"PPO training requires '{key}' in batch.")

        value_targets = batch.get("value_targets")
        return self.ppo_loss(
            task=batch["task"],
            variables=batch["variables"],
            actions=batch["actions"],
            old_log_probs=batch["old_log_probs"],
            advantages=batch["advantages"],
            clip_eps=clip_eps,
            value_targets=value_targets,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

    def _move_batch(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return tuple(
                value.to(device) if isinstance(value, Tensor) else value
                for value in batch
            )
        return batch

    @classmethod
    def build_optimizer(
        cls,
        model: "Head",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

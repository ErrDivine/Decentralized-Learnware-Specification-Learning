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
    """Head module producing an agent vote Pr(action=True|t,v)."""

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

        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, 1),
        )

    def _ensure_3d(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(1)
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def forward(
        self,
        task: Tensor,
        variables: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute logits for Pr(action=True|t,v)."""
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
        logits = self.head(pooled).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits

    def predict_vote(
        self,
        task: Tensor,
        variables: Tensor,
    ) -> Tensor:
        """Return probability vote in [0,1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(task, variables)
            return torch.sigmoid(logits)

    def should_act(
        self,
        task: Tensor,
        variables: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Return a boolean decision tensor based on threshold."""
        vote = self.predict_vote(task, variables)
        return vote >= threshold

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def training_step(self, batch: Union[Dict[str, Tensor], Sequence[Tensor]]) -> Tensor:
        if isinstance(batch, dict):
            task = batch.get("task")
            variables = batch.get("variables")
            targets = batch.get("targets")
        else:
            if len(batch) < 3:
                raise ValueError("batch must provide task, variables, and targets")
            task, variables, targets = batch[:3]

        logits = self.forward(task, variables)
        return self.compute_loss(logits, targets)

    def fit(
        self,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = 1.0,
    ) -> Sequence[float]:
        """Simple training loop for supervised vote labels."""
        self.train()
        losses = []
        for _ in range(epochs):
            for batch in dataloader:
                if device is not None:
                    batch = self._move_batch(batch, device)
                loss = self.training_step(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return losses

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

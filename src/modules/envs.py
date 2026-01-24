"""Environments for training heads with the Beta-policy PPO flow."""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from src.judger import judge_score
from src.agents.base_llm import BaseLLM


State = Tuple[Tensor, Tensor]  # (encoded task, encoded variables)


class DatasetHeadEnv:
    """Dataset-driven environment matching docs/theories.md.

    State_t = (t, v)
    Action: agent produces a vote; the highest vote agent acts.
    Transition: v <- v + delta_scale * (W_agent @ t)
    Reward: R = Δscore * vote_j, where score measures closeness of v to target variables.
    """

    def __init__(
        self,
        dataset: Sequence[dict],
        task_encoder: Callable[[dict], Tensor],
        variables_encoder: Callable[[dict], Tensor],
        n_agents: int,
        delta_scale: float = 0.05,
        max_steps: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.dataset = dataset
        self.task_encoder = task_encoder
        self.variables_encoder = variables_encoder
        self.n_agents = n_agents
        self.delta_scale = delta_scale
        self.max_steps = max_steps
        self.device = device

        # Infer dimensions from a sample
        sample = dataset[0]
        t_dim = int(task_encoder(sample).numel())
        v_dim = int(variables_encoder(sample).numel())

        self.agent_effects = torch.randn(n_agents, v_dim, t_dim, device=device)
        self.goal_scale = 1.0 / max(1, v_dim)
        self.reset()

    def reset(self, idx: Optional[int] = None) -> State:
        if idx is None:
            idx = random.randrange(len(self.dataset))
        sample = self.dataset[idx]
        self.raw_task = sample.get("question") or sample.get("problem") or ""
        self.raw_solution = sample.get("answer") or sample.get("solution") or ""

        self.t = self.task_encoder(sample).to(self.device)
        self.target_v = self.variables_encoder(sample).to(self.device)
        self.v = torch.zeros_like(self.target_v)
        self.score = self._judge(self.t, self.v)
        self.steps = 0
        return self.encoded_state

    @property
    def state(self) -> Tuple[str, str]:
        """Return raw task text and raw solution text (for LLM agents)."""
        return self.raw_task, self.raw_solution

    @property
    def encoded_state(self) -> State:
        return self.t, self.v

    @property
    def raw_state(self) -> Tuple[str, str]:
        """Return raw task text and current accumulated solution text (if available)."""
        return self.raw_task, self.raw_solution

    def _judge(self, t: Tensor, v: Tensor) -> float:
        # Higher score when v is close to target_v; tanh keeps it bounded.
        diff = v - self.target_v
        score = -torch.norm(diff) * self.goal_scale + 0.1 * t.mean()
        return float(torch.tanh(score))

    def step(self, agent_idx: int, vote: float) -> Tuple[float, State, float, bool]:
        delta_v = (self.agent_effects[agent_idx] @ self.t) * self.delta_scale
        self.v = self.v + delta_v
        new_score = self._judge(self.t, self.v)
        delta_score = new_score - self.score
        self.score = new_score
        self.steps += 1
        reward = delta_score * float(vote)
        done = self.steps >= self.max_steps
        return reward, self.encoded_state, delta_score, done


class TextEncoder:
    """Encodes text into a dense vector using a HF encoder model."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device or "cpu")
        self.device = device or self.model.device

    def encode(self, text: str) -> Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [1, seq, dim]
            mask = inputs.attention_mask.unsqueeze(-1)
            masked = last_hidden * mask
            summed = masked.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
        return mean_pooled.squeeze(0)


class LLMAgent:
    """Wrapper around a causal LLM to produce next-step text."""

    def __init__(self, model_path: str, system_prompt: str) -> None:
        self.llm = BaseLLM(model_path)
        self.system_prompt = system_prompt

    def act(self, task_text: str, history: str, max_new_tokens: int = 128) -> str:
        prompt = (
            f"Task:\\n{task_text}\\n\\n"
            f"Current solution (can be empty):\\n{history}\\n\\n"
            "Continue solving step by step. Provide the next step or refinement. "
            "Keep it concise."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.generate(messages, max_new_tokens=max_new_tokens)


class RealEnv:
    """Environment that drives real LLM agents and real reward via judger."""

    def __init__(
        self,
        dataset: Sequence[dict],
        encoder: TextEncoder,
        agents: Sequence[object],
        judger_model: str,
        judger_region: str,
        max_steps: int = 4,
        device: Optional[torch.device] = None,
        agent_max_new_tokens: int = 128,
    ) -> None:
        self.dataset = dataset
        self.encoder = encoder
        self.agents = agents
        self.judger_model = judger_model
        self.judger_region = judger_region
        self.max_steps = max_steps
        self.device = device or torch.device("cpu")
        self.agent_max_new_tokens = agent_max_new_tokens
        self.reset()

    def reset(self, idx: Optional[int] = None) -> State:
        if idx is None:
            idx = random.randrange(len(self.dataset))
        sample = self.dataset[idx]
        self.task_text = sample.get("question") or sample.get("problem") or ""
        self.solution_text = ""
        self.score = self._judge(self.task_text, self.solution_text)
        self.steps = 0
        return self.encoded_state

    @property
    def encoded_state(self) -> State:
        t_vec = self.encoder.encode(self.task_text).to(self.device)
        v_vec = self.encoder.encode(self.solution_text).to(self.device)
        return t_vec, v_vec

    def _judge(self, task: str, history: str) -> float:
        return float(
            judge_score(
                task=task,
                history=history,
                model=self.judger_model,
                region=self.judger_region,
            )
        )

    def step(self, agent_idx: int, vote: float) -> Tuple[float, State, float, bool]:
        agent = self.agents[agent_idx]
        new_text = self._agent_act(agent, self.task_text, self.solution_text)
        self.solution_text = (self.solution_text + "\\n" + new_text).strip()
        new_score = self._judge(self.task_text, self.solution_text)
        delta_score = new_score - self.score
        self.score = new_score
        self.steps += 1
        reward = delta_score * float(vote)
        done = self.steps >= self.max_steps
        return reward, self.encoded_state, delta_score, done

    def _agent_act(self, agent: object, task_text: str, history: str) -> str:
        if hasattr(agent, "act"):
            return agent.act(task_text, history, max_new_tokens=self.agent_max_new_tokens)  # type: ignore
        if hasattr(agent, "run"):
            prompt = (
                f"Task:\\n{task_text}\\n\\n"
                f"Current solution (can be empty):\\n{history}\\n\\n"
                "Continue solving step by step. Provide the next step or refinement. "
                "Keep it concise."
            )
            return agent.run(prompt, max_new_tokens=self.agent_max_new_tokens)  # type: ignore
        raise ValueError("Agent must implement act(task_text, history) or run(prompt)")

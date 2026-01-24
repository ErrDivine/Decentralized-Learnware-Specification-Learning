# Training and Inference Guide (Real-Mode, Paper-Ready)

This document specifies the end-to-end training and inference pipeline for the multi-agent Head system as implemented in this repository. All training is **real**: language-model agents generate steps, the judger provides rewards, and the Heads learn a Beta-policy over votes via PPO.

## Problem Setting
- State: `(t, v)` where `t` is the task text (problem statement) and `v` is the current solution/proof text (accumulated agent outputs).
- Agents: four fixed LLMs (Compute, Reason, Model, Verify) with their own system prompts and local model paths under `model/`.
- Head: one per agent, maps encoded `(t, v)` to Beta parameters `(α, β)`; the vote is the Beta mean (inference) or a sample (training).
- Selection: argmax over votes; only the winning agent acts.
- Reward: `R = Δscore * vote`, where `score` is returned by the judger on `(t, v)` (default `qwen-max`, region `beijing`).
- Termination: episode ends if `max_vote < stop_threshold` (early reset) or after `max_steps` actions.

## Architecture
- Encoder: `model/encoder` (HF model) via `TextEncoder` → mean-pooled embedding for `t` and `v`.
- Head (`src/modules/head.py`): MLP encoders for `t` and `v`, cross-attention fusion, FFN to 2-dim Beta params; optional critic head.
- Agents: `ComputeAgent`, `ReasonAgent`, `ModelAgent`, `VerifyAgent` (paths embedded; see `src/agents/*.py`).
- Environment: `RealEnv` (`src/modules/envs.py`) orchestrates agents, encoder, judger; applies termination logic.
- Judger: `src/judger.py`, uses DashScope API; requires `DASHSCOPE_API_KEY`.
- Training loop: `src/run.py` (real-only, no synthetic branches).
- Checkpoints: per-agent Head saved to `runs/heads_ckpt/head_agent{idx}_epoch{N}.pt`.

## Training Hyperparameters (fixed defaults in `src/run.py`)
- rollout_steps: 2048
- ppo_epochs: 12
- batch_size: 32
- lr: 3e-4
- clip_eps: 0.2
- value_coef: 0.5
- entropy_coef: 0.01
- hidden_dim: 512
- ff_dim: 1024
- max_steps (per episode): 8
- stop_threshold: 0.05 (early reset if all votes are small)
- save_every: 3 epochs
- device: `cuda` (recommended)
- agent_max_new_tokens: 256
- Agents: fixed to 4 (Compute, Reason, Model, Verify)

## Training Procedure
1) **Prerequisites**
   - Export `DASHSCOPE_API_KEY`.
   - Ensure GPU availability; install `accelerate` if you want auto device mapping.
   - Place models under `model/` (agents and encoder).
2) **Run training**
   ```bash
   python src/run.py \
     --dataset-path datasets/gsm8k_arrow \
     --split train \
     --device cuda \
     --save-dir runs/heads_ckpt
   ```
   - For hendrycks_math, change `--dataset-path datasets/hendrycks_math_arrow`.
3) **What happens each rollout step**
   - Encode `(t, v)` via `TextEncoder`.
   - Heads sample votes; if `max_vote < stop_threshold`, reset episode.
   - Argmax vote selects an agent; agent generates next solution text; judger scores new `(t, v)`; reward = `Δscore * vote`.
   - Store `(state, action, old_log_prob, advantage, value_target)` for the acting agent.
4) **Optimization**
   - PPO with clipping, entropy, and optional value loss (critic enabled).
   - Update only the acting agent’s Head.
5) **Checkpointing**
   - Saved every `save_every` epochs to `runs/heads_ckpt/`.

## Inference Procedure (multi-agent routing)
1) Load the latest checkpoints for all Heads and instantiate agents + encoder + judger.
2) Initialize `v = ""`; encode `(t, v)`; compute deterministic votes (`Head.predict_vote`).
3) Select agent with highest vote; agent generates next step; append to `v`.
4) Terminate when `max_vote < stop_threshold`, or after `max_steps`, or when an agent signals completion (recommended to add a completion check).
5) Return the final `v` as the solution trace.

## Evaluation / Single-Agent Baseline
- To compare, load a single Head and skip argmax; the loop reduces to one agent acting until termination.
- Use the same termination and judger scoring for fairness.

## Expected Outputs
- Checkpoints per agent in `runs/heads_ckpt/`.
- Logs: PPO losses per epoch, final deterministic votes on the current state.
- Solution traces are accumulated in `v` during rollout; you can log them for offline analysis.

## Failure Modes / Tips
- **OOM/timeout**: models are large; run on GPU, consider `accelerate` for sharding.
- **Missing key**: ensure `DASHSCOPE_API_KEY` is set.
- **Termination too early**: lower `stop_threshold` or increase `max_steps`.
- **Reward instability**: adjust `value_coef`/`entropy_coef` or increase `save_every` to reduce I/O.

## Scripts
- Training (multi-dataset): `scripts/train_real.sh` runs `src/run.py` on GSM8K and Hendrycks Math with GPU defaults.

## Reproducibility
- Seeds: `--seed` controls RNG for Heads and environment.
- Checkpoints: saved every `save_every` epochs with explicit epoch numbering.

## Narrative (paper-style)
We train four agent-specific Heads to model a Beta policy over action probabilities on math problem solving. Each episode starts from an empty solution trace `v` and a task `t`; Heads encode `(t, v)` via a shared text encoder and produce Beta parameters whose mean is the agent’s vote. The highest-vote agent generates the next CoT step with its LLM; a judger scores progress, yielding reward `R = Δscore · vote`. We collect per-agent trajectories and optimize only the acting agent’s Head with PPO (clipped objective, entropy regularization, and a critic). Episodes terminate when the maximal vote falls below a threshold or when a fixed step budget is reached, encouraging Heads to learn when to abstain. This loop runs over real math datasets (e.g., GSM8K, Hendrycks Math) with fixed hyperparameters and periodic checkpointing.

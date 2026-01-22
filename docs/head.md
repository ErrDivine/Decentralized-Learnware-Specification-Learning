# Head Module

The head maps the shared state `(t, v)` to a *Beta* distribution over vote probabilities for one agent. During execution each agent holds its own `Head` instance (with its own parameters), so no agent ID is encoded inside the model.

## Inputs and outputs
- Inputs: `task` tensor and `variables` tensor. Shapes can be `[D]`, `[B, D]`, or `[B, L, D]` and are normalized to sequences inside the module.
- Output (policy): `(alpha, beta)` parameters of a Beta distribution, shape `[B, 2]`. Use `sample_vote()` for stochastic votes or `predict_vote()`/`distribution().mean` for deterministic votes.
- Critic (optional): if `use_critic=True` in `HeadConfig`, `critic_value(task, variables)` predicts a scalar value per sample.

## Architecture (per docs/theories.md)
```
t --MLP--> t'  \
 +            cross-attn --> pooled --> policy MLP --> alpha,beta
v --MLP--> v'  /
```
- Separate MLP encoders for heterogeneous task and variables vectors.
- Cross-attention fuses `t'` (query) with `v'` (key/value).
- Policy head outputs Beta params via `softplus` (kept >0).
- Optional critic shares encoders but has its own MLP head.

## Key APIs
- `HeadConfig(task_dim, var_dim, hidden_dim=256, attention_heads=4, ff_dim=256, encoder_depth=2, dropout=0.1, use_critic=True)`
- `forward(task, variables) -> alpha_beta`
- `distribution(task, variables) -> torch.distributions.Beta`
- `sample_vote(task, variables, deterministic=False, return_aux=False)`  
  Returns vote in `[0,1]` and, if `return_aux`, also `log_prob`, `alpha`, `beta`. Use `deterministic=True` for eval (mean of Beta).
- `predict_vote(task, variables)`  
  Deterministic vote (Beta mean), convenience for inference/routing.
- `should_act(task, variables, threshold=0.5)`  
  Boolean decision.
- `critic_value(task, variables)`  
  Requires `use_critic=True`.
- `ppo_loss(task, variables, actions, old_log_probs, advantages, clip_eps=0.2, value_targets=None, value_coef=0.5, entropy_coef=0.0)`  
  Implements the PPO objective from docs/theories.md on the Beta policy; returns `(loss, stats)`.
- `fit(...)`  
  Supports `supervised=True` (legacy NLL on provided targets) or `supervised=False` to consume PPO-style batches (delegates to `ppo_loss`).

## PPO batch contract (per Buffer in docs/theories.md)
For PPO training (`supervised=False`), each batch must provide:
- `task`: tensor `[B, ...]`
- `variables`: tensor `[B, ...]`
- `actions`: sampled votes from the old policy `[B]` or `[B, L]`, clamped to `(0,1)`
- `old_log_probs`: log-prob of those actions under the *old* Beta policy `[B]`
- `advantages`: advantage estimates `[B]`
- Optional: `value_targets` `[B]` if training the critic

The loss computes `ratio = exp(logP_new - logP_old)`, applies PPO clipping, adds optional value loss, and entropy regularization.

## Usage sketches
```python
from src.modules.head import Head, HeadConfig

head = Head(HeadConfig(task_dim=5, var_dim=5, use_critic=True))

# rollout (training)
vote, logp, alpha, beta = head.sample_vote(task, variables, return_aux=True)
buffer.append({"task": task, "variables": variables,
               "actions": vote.detach(),
               "old_log_probs": logp.detach(),
               "advantages": adv, "value_targets": vt})

# update
loss, stats = head.ppo_loss(task=batch["task"],
                            variables=batch["variables"],
                            actions=batch["actions"],
                            old_log_probs=batch["old_log_probs"],
                            advantages=batch["advantages"],
                            value_targets=batch.get("value_targets"))
loss.backward()
optimizer.step()

# inference/routing
vote = head.predict_vote(task, variables)
```

## Notes
- Each agent should instantiate its own `Head`; parameter divergence encodes agent-specific behavior.
- `alpha/beta` are offset by `1e-4` after `softplus` to avoid zero concentration.
- If you do not need value baselines, set `use_critic=False` and omit `value_targets`.
- For supervised pretraining, provide `targets` in batches; the model minimizes negative log-prob under the Beta policy.

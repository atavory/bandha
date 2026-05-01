"""BANDHA demo: train a small model and evaluate on synthetic bandits."""

import torch
import numpy as np
from envs import make_env
from model import BanditPFN
from losses import loss_fn, bandit_loss_fn

print("=== BANDHA Demo ===\n")

# Train a small model
K, d, T = 3, 3, 200
torch.manual_seed(42)
model = BanditPFN(d_ctx=d, K=K, d_model=64, n_layers=2, n_heads=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
rng = np.random.default_rng(42)

print(f"Training: K={K}, d={d}, T={T}, formula-only prior")
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params\n")

for step in range(200):
    envs = [make_env(K=K, d=d, n=T, rng=rng, prior_type="formula") for _ in range(8)]
    ctx = torch.stack([e.contexts for e in envs])
    am = torch.stack([e.arm_means for e in envs])
    arms = torch.stack([e.true_arms for e in envs])

    result = loss_fn(model, ctx, am, arms)
    optimizer.zero_grad()
    result.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        print(f"  step {step:3d}: acc={result.metrics['accuracy']:.1%} "
              f"norm={result.metrics['regret_norm']:.3f}")

# Evaluate
print("\n=== Online Bandit Eval ===\n")
result = bandit_loss_fn(model, n_envs=20, K=K, d=d, T=T, n_seeds=1,
                         device=torch.device("cpu"), prior_type="formula")

for name in result.regret_table:
    regrets = [np.mean(result.regret_table[name][t]) for t in result.checkpoints]
    print(f"  {name:>12s}: " + "  ".join(f"T={t}:{r:.1f}" for t, r in zip(result.checkpoints, regrets)))

print("\nDone.")

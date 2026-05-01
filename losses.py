"""Loss functions for BanditPFN.

loss_fn:        teacher-forced on logged histories. Used by train AND eval mode 1.
bandit_loss_fn: online sequential rollout vs baselines. Used by eval mode 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from envs import BanditEnv, make_env
from model import BanditPFN


@dataclass
class LossResult:
    loss: torch.Tensor
    metrics: dict[str, float]


def build_reward_history(
    actions: torch.Tensor,
    rewards: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Build shifted feedback vectors from logged actions and rewards.

    Args:
        actions: [batch, T] long
        rewards: [batch, T] float (0 or 1)
        K: number of arms

    Returns:
        reward_vecs: [batch, T, K] — shifted by 1, encoding:
            0 = not pulled, 1 = pulled wrong, 2 = pulled correct
    """
    batch, T = actions.shape
    reward_vecs = torch.zeros(batch, T, K, device=actions.device)
    for t in range(T - 1):
        reward_vecs[torch.arange(batch), t + 1, actions[:, t]] = rewards[:, t] + 1.0
    return reward_vecs


def sample_logged_history(
    true_arms: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random logged trajectory.

    Returns:
        actions: [batch, T] — random arm pulls
        rewards: [batch, T] — 1 if correct, 0 if wrong
    """
    batch, T = true_arms.shape
    actions = torch.randint(0, K, (batch, T), device=true_arms.device)
    rewards = (actions == true_arms).float()
    return actions, rewards


def loss_fn(
    model: BanditPFN,
    contexts: torch.Tensor,
    arm_means: torch.Tensor,
    true_arms: torch.Tensor,
    loss_start_frac: float = 0.25,
) -> LossResult:
    """Training loss: CE on soft arm targets from logged histories.

    1. Generate random logged history (random arm pulls)
    2. Build shifted feedback vectors
    3. One forward pass with causal mask
    4. CE against soft arm_means, skipping early positions

    Args:
        model:           BanditPFN
        contexts:        [batch, T, d]
        arm_means:       [batch, T, K] soft arm probabilities
        true_arms:       [batch, T] long — sampled correct arm
        loss_start_frac: skip this fraction of early positions in the loss
    """
    K = model.K
    actions, rewards = sample_logged_history(true_arms, K)
    reward_vecs = build_reward_history(actions, rewards, K)

    logits = model(contexts, reward_vecs)  # [batch, T, K]

    # CE against soft arm targets, late positions only
    T = contexts.shape[1]
    start = max(int(T * loss_start_frac), 1)
    per_step_loss = -(arm_means * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss = per_step_loss[:, start:].mean()

    # metrics
    split = T // 2
    tail = max(T // 4, 1)
    with torch.no_grad():
        chosen = logits.argmax(dim=-1)
        best = arm_means.argmax(dim=-1)
        chosen_means = arm_means.gather(2, chosen.unsqueeze(-1)).squeeze(-1)
        optimal_means = arm_means.max(dim=-1).values
        soft_regret = optimal_means - chosen_means
        random_regret = float((optimal_means - arm_means.mean(dim=-1)).mean().item())

    return LossResult(
        loss=loss,
        metrics={
            "regret_early": soft_regret[:, :split].mean().item(),
            "regret_late": soft_regret[:, split:].mean().item(),
            "accuracy": (chosen == best).float().mean().item(),
            "accuracy_final20": (chosen[:, -20:] == best[:, -20:]).float().mean().item(),
            "random_regret": max(random_regret, 1e-8),
            "regret_norm": float(soft_regret.mean().item()) / max(random_regret, 1e-8),
        },
    )


# ---------------------------------------------------------------------------
# Bandit eval: online sequential rollout vs baselines
# ---------------------------------------------------------------------------

def random_policy(history, ctx, K):
    return int(np.random.randint(K))


def epsilon_greedy(eps=0.1):
    def policy(history, ctx, K):
        if np.random.random() < eps or not history:
            return int(np.random.randint(K))
        arm_rewards = {}
        for _, a, r in history:
            arm_rewards.setdefault(a, []).append(r)
        means = [np.mean(arm_rewards.get(k, [0.0])) for k in range(K)]
        return int(np.argmax(means))
    return policy


def lin_ucb(alpha=1.0):
    def policy(history, ctx, K):
        d = len(ctx)
        A = [np.eye(d) for _ in range(K)]
        b = [np.zeros(d) for _ in range(K)]
        for x, a, r in history:
            A[a] += np.outer(x, x)
            b[a] += r * x
        vals = []
        for k in range(K):
            A_inv = np.linalg.solve(A[k], np.eye(d))
            theta = A_inv @ b[k]
            vals.append(ctx @ theta + alpha * np.sqrt(ctx @ A_inv @ ctx))
        return int(np.argmax(vals))
    return policy


def lin_ts(v_sq=1.0):
    def policy(history, ctx, K):
        d = len(ctx)
        A = [np.eye(d) for _ in range(K)]
        b = [np.zeros(d) for _ in range(K)]
        for x, a, r in history:
            A[a] += np.outer(x, x)
            b[a] += r * x
        samples = []
        for k in range(K):
            A_inv = np.linalg.solve(A[k], np.eye(d))
            theta_hat = A_inv @ b[k]
            samples.append(ctx @ np.random.multivariate_normal(theta_hat, v_sq * A_inv))
        return int(np.argmax(samples))
    return policy


def make_pfn_policy(model, device):
    def policy(history, ctx, K):
        T = len(history) + 1
        d = len(ctx)
        contexts = torch.zeros(1, T, model.d_ctx, device=device)
        reward_vecs = torch.zeros(1, T, model.K, device=device)
        for i, (x, a, r) in enumerate(history):
            contexts[0, i, :d] = torch.tensor(x, dtype=torch.float32)
            if i + 1 < T:
                reward_vecs[0, i + 1, a] = r + 1
        contexts[0, T - 1, :d] = torch.tensor(ctx, dtype=torch.float32)
        return model.select_arm(contexts, reward_vecs, T - 1).item()
    return policy


@dataclass
class BanditResult:
    regret_table: dict[str, dict[int, list]]
    checkpoints: list[int]


def bandit_loss_fn(
    model: BanditPFN,
    n_envs: int,
    K: int,
    d: int,
    T: int,
    n_seeds: int,
    device: torch.device,
    prior_type: str = "formula",
) -> BanditResult:
    model.eval()
    checkpoints = [t for t in [10, 20, 50, 100, 200, 500] if t <= T]
    if not checkpoints:
        checkpoints = [T]

    policies = {
        "PFN": make_pfn_policy(model, device),
        "LinTS": lin_ts(),
        "LinUCB": lin_ucb(),
        "EpsGreedy": epsilon_greedy(),
        "Random": random_policy,
    }
    all_results = {name: {t: [] for t in checkpoints} for name in policies}

    for env_id in range(n_envs):
        rng = np.random.default_rng(env_id * 1000)
        env = make_env(K=K, d=d, n=T, rng=rng, prior_type=prior_type)
        for seed in range(n_seeds):
            for pi, (name, fn) in enumerate(policies.items()):
                np.random.seed(env_id * 100000 + seed * 100 + pi)
                torch.manual_seed(env_id * 100000 + seed * 100 + pi)
                history = []
                cum_regret = 0.0
                results = {}
                for t in range(T):
                    ctx = env.contexts[t].numpy()
                    arm = fn(history, ctx, env.K)
                    reward = env.step(t, arm)
                    cum_regret += env.regret(t, arm)
                    history.append((ctx, arm, reward))
                    if t + 1 in checkpoints:
                        results[t + 1] = cum_regret
                for t in checkpoints:
                    all_results[name][t].append(results[t])

    return BanditResult(regret_table=all_results, checkpoints=checkpoints)

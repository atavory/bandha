"""Fixed-batch overfit gate for BANDHA model variants."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from band_pfn.algs.envs import make_env
from band_pfn.algs.losses import LossResult, build_reward_history
from band_pfn.algs.model import BanditPFN
from band_pfn.algs.model_perarm import BanditPFNPerArm


def make_fixed_batch(
    *,
    n_envs: int,
    K: int,
    d: int,
    T: int,
    seed: int,
    prior_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    ctxs, arm_means, true_arms = [], [], []
    for _ in range(n_envs):
        env = make_env(K=K, d=d, n=T, rng=rng, prior_type=prior_type)
        ctxs.append(env.contexts)
        arm_means.append(env.arm_means)
        true_arms.append(env.true_arms)
    return torch.stack(ctxs), torch.stack(arm_means), torch.stack(true_arms)


def build_model(args: argparse.Namespace, device: torch.device):
    model_cls = BanditPFN if args.model_arch == "v9" else BanditPFNPerArm
    return model_cls(
        d_ctx=args.d,
        K=args.K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=0.0,
        max_T=max(args.T, 128),
    ).to(device)


def fixed_logged_loss(
    model,
    contexts: torch.Tensor,
    arm_means: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    loss_start_frac: float = 0.25,
) -> LossResult:
    reward_vecs = build_reward_history(actions, rewards, model.K)
    logits = model(contexts, reward_vecs)

    T = contexts.shape[1]
    start = max(int(T * loss_start_frac), 1)
    per_step_loss = -(arm_means * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss = per_step_loss[:, start:].mean()

    split = T // 2
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
            "accuracy_final20": (chosen[:, -20:] == best[:, -20:])
            .float()
            .mean()
            .item(),
            "random_regret": max(random_regret, 1e-8),
            "regret_norm": float(soft_regret.mean().item()) / max(random_regret, 1e-8),
        },
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", choices=["v9", "perarm"], default="perarm")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--T", type=int, default=80)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--prior-type", type=str, default="formula")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.manual_seed(args.seed)
    contexts, arm_means, true_arms = make_fixed_batch(
        n_envs=args.n_envs,
        K=args.K,
        d=args.d,
        T=args.T,
        seed=args.seed,
        prior_type=args.prior_type,
    )
    contexts = contexts.to(device)
    arm_means = arm_means.to(device)
    true_arms = true_arms.to(device)
    actions = torch.randint(args.K, true_arms.shape, device=device)
    rewards = (actions == true_arms).float()

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    first = None
    last = None
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = fixed_logged_loss(model, contexts, arm_means, actions, rewards)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if first is None:
            first = result
        last = result
        print(
            f"{step:02d} loss={result.loss.item():.4f} "
            f"acc={result.metrics['accuracy']:.1%} "
            f"f20={result.metrics['accuracy_final20']:.1%} "
            f"norm={result.metrics['regret_norm']:.3f}"
        )

    assert first is not None and last is not None
    print(
        "delta "
        f"loss={first.loss.item() - last.loss.item():.4f} "
        f"norm={first.metrics['regret_norm'] - last.metrics['regret_norm']:.3f}"
    )


if __name__ == "__main__":
    main()

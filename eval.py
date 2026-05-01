"""BanditPFN evaluation.

Two modes:
  --mode train-loss   Same loss_fn as training on held-out logged histories.
  --mode bandit       Online sequential rollout vs baselines.

Usage:
    python3 -m band_pfn.algs.eval --checkpoint X.pt --mode train-loss
    python3 -m band_pfn.algs.eval --checkpoint X.pt --mode bandit --T 200
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from envs import make_env
from losses import bandit_loss_fn, loss_fn
from model import BanditPFN


@torch.no_grad()
def run_train_loss_eval(model, n_envs, K, d, T, device, rng, prior_type):
    model.eval()
    all_metrics: dict[str, list] = {}
    for _ in range(n_envs):
        env = make_env(K=K, d=d, n=T, rng=rng, prior_type=prior_type)
        result = loss_fn(
            model,
            env.contexts.unsqueeze(0).to(device),
            env.arm_means.unsqueeze(0).to(device),
            env.true_arms.unsqueeze(0).to(device),
        )
        for k, v in result.metrics.items():
            all_metrics.setdefault(k, []).append(v)

    m = {k: np.mean(v) for k, v in all_metrics.items()}
    print(
        f"\n=== Train-loss eval ({n_envs} envs, T={T}, K={K}, prior={prior_type}) ==="
    )
    print(f"  Accuracy:       {m['accuracy']:.1%}  (random = {1 / K:.1%})")
    print(f"  Accuracy F20:   {m['accuracy_final20']:.1%}")
    print(f"  Regret norm:    {m['regret_norm']:.3f}  (1.0 = random)")
    print(f"  Regret early:   {m['regret_early']:.4f}")
    print(f"  Regret late:    {m['regret_late']:.4f}")


def run_bandit_eval(model, n_envs, K, d, T, n_seeds, device, prior_type):
    result = bandit_loss_fn(model, n_envs, K, d, T, n_seeds, device, prior_type)
    cp = result.checkpoints

    print(
        f"\n=== Bandit eval ({n_envs} envs, T={T}, K={K}, {n_seeds} seeds, prior={prior_type}) ==="
    )
    print(f"{'Method':>12s}", end="")
    for t in cp:
        print(f"  T={t:>4d}", end="")
    print()
    print("-" * (12 + 8 * len(cp)))

    for name in result.regret_table:
        print(f"{name:>12s}", end="")
        for t in cp:
            print(f"  {np.mean(result.regret_table[name][t]):6.1f}", end="")
        print()

    last_t = cp[-1]
    pfn = result.regret_table["PFN"]
    for bl in ["LinTS", "LinUCB", "Random"]:
        wins = sum(
            1 for a, b in zip(pfn[last_t], result.regret_table[bl][last_t]) if a < b
        )
        total = len(pfn[last_t])
        print(f"  PFN vs {bl}: {wins}/{total} wins ({wins / total:.0%})")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["train-loss", "bandit"], default="bandit")
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--prior-type", type=str, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ca = ckpt["args"]

    model = BanditPFN(
        d_ctx=ca["d"],
        K=ca["K"],
        d_model=ca["d_model"],
        n_heads=ca["n_heads"],
        n_layers=ca["n_layers"],
        ff_mult=ca.get("ff_mult", 4),
        dropout=ca.get("dropout", 0.0),
        max_T=ca.get("max_T", max(ca["T"], 1024)),
        activation_checkpointing=ca.get("activation_checkpointing", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prior_type = args.prior_type or ca.get("prior_type", "formula")
    print(f"Loaded {args.checkpoint} (K={ca['K']}, d={ca['d']}, prior={prior_type})")

    if args.mode == "train-loss":
        rng = np.random.default_rng(99)
        run_train_loss_eval(
            model, args.n_envs, ca["K"], ca["d"], args.T, device, rng, prior_type
        )
    else:
        run_bandit_eval(
            model,
            args.n_envs,
            ca["K"],
            ca["d"],
            args.T,
            args.n_seeds,
            device,
            prior_type,
        )


if __name__ == "__main__":
    main()

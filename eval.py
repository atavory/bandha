"""BanditPFN evaluation.

Two modes:
  --mode train-loss   Same loss_fn as training on held-out logged histories.
  --mode bandit       Online sequential rollout vs baselines.

Usage:
    python3 eval.py --checkpoint X.pt --mode train-loss
    python3 eval.py --checkpoint X.pt --mode bandit --T 200
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from envs import make_env
from losses import bandit_loss_fn, handoff_bandit_loss_fn, loss_fn
from model import BanditPFN
from model_perarm import BanditPFNPerArm


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


def parse_handoffs(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def run_handoff_eval(
    model,
    n_envs,
    K,
    d,
    T,
    n_seeds,
    device,
    prior_type,
    handoff_ats,
):
    result = handoff_bandit_loss_fn(
        model,
        n_envs,
        K,
        d,
        T,
        n_seeds,
        device,
        handoff_ats=handoff_ats,
        prior_type=prior_type,
    )
    cp = result.checkpoints

    print(
        f"\n=== Handoff eval ({n_envs} envs, T={T}, K={K}, {n_seeds} seeds, prior={prior_type}) ==="
    )
    print(f"{'Method':>18s}", end="")
    for t in cp:
        print(f"  T={t:>4d}", end="")
    print()
    print("-" * (18 + 8 * len(cp)))

    for name in result.regret_table:
        print(f"{name:>18s}", end="")
        for t in cp:
            print(f"  {np.mean(result.regret_table[name][t]):6.1f}", end="")
        print()

    last_t = cp[-1]
    for name in result.regret_table:
        if not name.startswith("PFN->"):
            continue
        for bl in ["LinTS", "LinUCB"]:
            wins = sum(
                1
                for a, b in zip(
                    result.regret_table[name][last_t],
                    result.regret_table[bl][last_t],
                )
                if a < b
            )
            total = len(result.regret_table[name][last_t])
            print(f"  {name} vs {bl}: {wins}/{total} wins ({wins / total:.0%})")


def build_model_from_checkpoint(
    checkpoint_args: dict,
    *,
    K: int | None,
    device: torch.device,
) -> torch.nn.Module:
    model_arch = checkpoint_args.get("model_arch", "v9")
    model_cls = BanditPFNPerArm if model_arch == "perarm" else BanditPFN
    model_k = int(K if K is not None else checkpoint_args["K"])
    if model_arch != "perarm" and model_k != int(checkpoint_args["K"]):
        raise ValueError(
            f"Fixed-K checkpoint has K={checkpoint_args['K']} but eval requested K={model_k}"
        )
    return model_cls(
        d_ctx=checkpoint_args["d"],
        K=model_k,
        d_model=checkpoint_args["d_model"],
        n_heads=checkpoint_args["n_heads"],
        n_layers=checkpoint_args["n_layers"],
        ff_mult=checkpoint_args.get("ff_mult", 4),
        dropout=checkpoint_args.get("dropout", 0.0),
        max_T=checkpoint_args.get("max_T", max(checkpoint_args["T"], 1024)),
        activation_checkpointing=checkpoint_args.get("activation_checkpointing", False),
    ).to(device)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=["train-loss", "bandit", "handoff"],
        default="bandit",
    )
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--handoff-ats", type=str, default="50,100,200")
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

    eval_k = int(args.K if args.K is not None else ca["K"])
    model = build_model_from_checkpoint(ca, K=eval_k, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prior_type = args.prior_type or ca.get("prior_type", "formula")
    print(
        f"Loaded {args.checkpoint} "
        f"(train K={ca['K']}, eval K={eval_k}, d={ca['d']}, prior={prior_type})"
    )

    if args.mode == "train-loss":
        rng = np.random.default_rng(99)
        run_train_loss_eval(
            model, args.n_envs, eval_k, ca["d"], args.T, device, rng, prior_type
        )
    elif args.mode == "bandit":
        run_bandit_eval(
            model,
            args.n_envs,
            eval_k,
            ca["d"],
            args.T,
            args.n_seeds,
            device,
            prior_type,
        )
    else:
        run_handoff_eval(
            model,
            args.n_envs,
            eval_k,
            ca["d"],
            args.T,
            args.n_seeds,
            device,
            prior_type,
            parse_handoffs(args.handoff_ats),
        )


if __name__ == "__main__":
    main()

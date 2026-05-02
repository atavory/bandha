"""Evaluate BANDHA on real-world classification datasets adapted to bandits.

Each multiclass dataset becomes a contextual bandit:
  - context = feature vector (standardized)
  - K arms = K classes
  - reward = 1 if selected arm matches true class, 0 otherwise
  - features projected to d_model dimensions via PCA if d > d_model

Usage:
    python3 scripts/eval_realworld.py \
        --checkpoint path/to/checkpoint.pt \
        --dataset covertype \
        --T 200 --n-seeds 3 \
        --output results/realworld/covertype.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from band_pfn.algs.losses import (
    BanditResult,
    epsilon_greedy,
    lin_ts,
    lin_ucb,
    make_checkpoints,
    make_pfn_policy,
    random_policy,
    rollout_policy,
)
from band_pfn.algs.envs import BanditEnv
from band_pfn.algs.model import BanditPFN


@dataclass
class DatasetInfo:
    openml_id: int
    n_classes: int


DATASETS = {
    "iris": DatasetInfo(openml_id=61, n_classes=3),
    "wine": DatasetInfo(openml_id=187, n_classes=3),
    "balance-scale": DatasetInfo(openml_id=11, n_classes=3),
    "cmc": DatasetInfo(openml_id=23, n_classes=3),
    "page-blocks": DatasetInfo(openml_id=30, n_classes=5),
    "optdigits": DatasetInfo(openml_id=28, n_classes=10),
    "yeast": DatasetInfo(openml_id=181, n_classes=10),
    "vehicle": DatasetInfo(openml_id=54, n_classes=4),
    "segment": DatasetInfo(openml_id=36, n_classes=7),
    "satimage": DatasetInfo(openml_id=182, n_classes=6),
    "pendigits": DatasetInfo(openml_id=32, n_classes=10),
    "covertype": DatasetInfo(openml_id=44121, n_classes=7),
    "letter": DatasetInfo(openml_id=6, n_classes=26),
    "volkert": DatasetInfo(openml_id=41166, n_classes=10),
    "jannis": DatasetInfo(openml_id=41168, n_classes=4),
    "helena": DatasetInfo(openml_id=41169, n_classes=100),
}


def load_real_dataset(
    name: str,
    data_home: str = "/tmp/sklearn_data",
) -> tuple[np.ndarray, np.ndarray, int]:
    info = DATASETS[name]
    print(f"Loading {name} (OpenML {info.openml_id})...", end="", flush=True)
    data = fetch_openml(
        data_id=info.openml_id, as_frame=True, parser="auto", data_home=data_home
    )
    df = data.frame
    target_col = data.target.name if hasattr(data.target, "name") else "target"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]

    y_raw = data.target.values[mask]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    K = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f" {X.shape[0]} samples, {X.shape[1]} features, {K} classes")
    return X, y, K


def make_realworld_env(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    T: int,
    d: int,
    rng: np.random.Generator,
    pca: PCA | None = None,
) -> BanditEnv:
    indices = rng.choice(len(X), size=T, replace=len(X) < T)
    contexts_np = X[indices]
    labels = y[indices]

    if pca is not None:
        contexts_np = pca.transform(contexts_np)
    elif contexts_np.shape[1] > d:
        contexts_np = contexts_np[:, :d]
    elif contexts_np.shape[1] < d:
        pad = np.zeros((T, d - contexts_np.shape[1]), dtype=np.float32)
        contexts_np = np.concatenate([contexts_np, pad], axis=1)

    contexts = torch.from_numpy(contexts_np.astype(np.float32))
    true_arms = torch.from_numpy(labels.astype(np.int64))

    arm_means = torch.zeros(T, K)
    for t in range(T):
        arm_means[t, true_arms[t]] = 1.0

    return BanditEnv(
        contexts=contexts,
        arm_means=arm_means,
        true_arms=true_arms,
        K=K,
    )


def run_realworld_eval(
    model: BanditPFN,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    d: int,
    T: int,
    n_envs: int,
    n_seeds: int,
    device: torch.device,
    pca: PCA | None = None,
) -> BanditResult:
    model.eval()
    checkpoints = make_checkpoints(T)

    policies = {
        "PFN": make_pfn_policy(model, device),
        "LinTS": lin_ts(),
        "LinUCB": lin_ucb(),
        "EpsGreedy": epsilon_greedy(),
        "Random": random_policy,
    }
    all_results = {name: {t: [] for t in checkpoints} for name in policies}

    t0 = time.time()
    for env_id in range(n_envs):
        rng = np.random.default_rng(env_id * 1000)
        env = make_realworld_env(X, y, K, T, d, rng, pca)
        for seed in range(n_seeds):
            for pi, (name, fn) in enumerate(policies.items()):
                np.random.seed(env_id * 100000 + seed * 100 + pi)
                torch.manual_seed(env_id * 100000 + seed * 100 + pi)
                results = rollout_policy(env, fn, T=T, checkpoints=checkpoints)
                for t in checkpoints:
                    all_results[name][t].append(results[t])
        if (env_id + 1) % 10 == 0 or env_id == 0:
            elapsed = time.time() - t0
            eta = elapsed / (env_id + 1) * (n_envs - env_id - 1)
            print(
                f"  {env_id + 1}/{n_envs} envs, "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                flush=True,
            )

    return BanditResult(regret_table=all_results, checkpoints=checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    X, y, K_data = load_real_dataset(args.dataset)

    if K_data != ca["K"]:
        print(
            f"WARNING: dataset has {K_data} classes but model trained on K={ca['K']}. "
            f"Skipping — need a model with matching K."
        )
        return

    d_model_ctx = ca["d"]
    pca = None
    if args.use_pca and X.shape[1] > d_model_ctx:
        pca = PCA(n_components=d_model_ctx)
        pca.fit(X)
        print(f"PCA: {X.shape[1]} -> {d_model_ctx} dims")

    result = run_realworld_eval(
        model, X, y, K_data, d_model_ctx, args.T,
        args.n_envs, args.n_seeds, device, pca,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cp = result.checkpoints
    print(f"\n=== {args.dataset} ({K_data} classes, T={args.T}) ===")
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

    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method"] + [f"T={t}" for t in cp])
        for name in result.regret_table:
            row = [args.dataset, name]
            for t in cp:
                row.append(f"{np.mean(result.regret_table[name][t]):.1f}")
            w.writerow(row)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()

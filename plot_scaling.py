"""Plot model scaling: regret vs model size.

Usage:
    python3 scripts/plot_scaling.py --output figures/scaling.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCALING_DATA = [
    {"label": "v9 formula 200ep", "params": 100_611, "regret_200": 56.8, "win_linucb": 57},
    {"label": "v9 formula 400ep", "params": 100_611, "regret_200": 54.0, "win_linucb": 57},
    {"label": "v9 mixed scratch", "params": 100_611, "regret_200": 44.6, "win_linucb": 80},
    {"label": "v9 mixed ws", "params": 100_611, "regret_200": 42.6, "win_linucb": 88},
    {"label": "v10 HUGE mixed", "params": 37_800_000, "regret_200": 33.3, "win_linucb": 96},
]

BASELINES = {
    "LinUCB": 56.8,
    "LinTS": 65.6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/scaling.pdf")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    params = [d["params"] for d in SCALING_DATA]
    regrets = [d["regret_200"] for d in SCALING_DATA]
    wins = [d["win_linucb"] for d in SCALING_DATA]
    labels = [d["label"] for d in SCALING_DATA]

    ax1.scatter(params, regrets, c="black", s=40, zorder=3)
    for p, r, lbl in zip(params, regrets, labels):
        ax1.annotate(lbl, (p, r), fontsize=6, textcoords="offset points",
                     xytext=(5, 5))
    for bl_name, bl_val in BASELINES.items():
        ax1.axhline(bl_val, color="gray", linestyle="--", linewidth=0.8)
        ax1.text(max(params) * 0.5, bl_val + 1, bl_name, fontsize=7, color="gray")
    ax1.set_xscale("log")
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Cumulative regret at $T=200$")
    ax1.set_title("Regret vs. model scale")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.scatter(params, wins, c="black", s=40, zorder=3)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax2.text(min(params) * 2, 51, "50% = tied", fontsize=7, color="gray")
    ax2.set_xscale("log")
    ax2.set_xlabel("Parameters")
    ax2.set_ylabel("Win rate vs. LinUCB (%)")
    ax2.set_title("Win rate vs. model scale")
    ax2.set_ylim(40, 100)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


if __name__ == "__main__":
    main()

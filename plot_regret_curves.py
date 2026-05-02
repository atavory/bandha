"""Plot regret curves from parsed eval CSVs.

Usage:
    python3 scripts/plot_regret_curves.py \
        --csvs results/parsed/k3_mixed/eval_bandit_mixed_k3d3_warmstart.csv \
        --title "K=3 Mixed" \
        --output figures/regret_k3_mixed.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "PFN": {"color": "black", "linewidth": 2.0, "marker": "o", "markersize": 5},
    "LinTS": {"color": "#666666", "linewidth": 1.5, "marker": "s", "markersize": 4, "linestyle": "--"},
    "LinUCB": {"color": "#999999", "linewidth": 1.5, "marker": "^", "markersize": 4, "linestyle": "-."},
    "EpsGreedy": {"color": "#bbbbbb", "linewidth": 1.0, "marker": "d", "markersize": 3, "linestyle": ":"},
    "Random": {"color": "#cccccc", "linewidth": 1.0, "marker": "x", "markersize": 3, "linestyle": ":"},
}


def load_regret_csv(path: Path) -> dict[str, list[tuple[int, float]]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        horizons = [int(h.replace("T=", "")) for h in header[1:]]
        data = {}
        for row in reader:
            name = row[0]
            values = [float(v) for v in row[1:]]
            data[name] = list(zip(horizons, values))
    return data


def plot_regret(
    data: dict[str, list[tuple[int, float]]],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for name, points in data.items():
        style = STYLE.get(name, {"color": "gray", "linewidth": 1.0})
        ts = [t for t, _ in points]
        vs = [v for _, v in points]
        ax.plot(ts, vs, label=name, **style)

    ax.set_xlabel("Horizon $T$")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--title", default="Regret Curves")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_data = {}
    for csv_path in args.csvs:
        data = load_regret_csv(Path(csv_path))
        all_data.update(data)

    plot_regret(all_data, args.title, Path(args.output))


if __name__ == "__main__":
    main()

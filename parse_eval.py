"""Parse bandit eval .txt files into structured CSVs.

Usage:
    python3 scripts/parse_eval.py results/ --output results/parsed/
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_bandit_eval(text: str) -> dict:
    result = {"header": "", "methods": {}, "wins": {}}
    lines = text.strip().split("\n")

    for line in lines:
        if line.startswith("==="):
            result["header"] = line.strip("= \n")
            continue
        if line.startswith("---"):
            continue

        win_match = re.match(r"\s+(\S+) vs (\S+): (\d+)/(\d+) wins \((\d+)%\)", line)
        if win_match:
            key = f"{win_match.group(1)}_vs_{win_match.group(2)}"
            result["wins"][key] = {
                "wins": int(win_match.group(3)),
                "total": int(win_match.group(4)),
                "pct": int(win_match.group(5)),
            }
            continue

        parts = line.split()
        if len(parts) >= 2 and not parts[0].startswith("Method"):
            name = parts[0]
            values = []
            for p in parts[1:]:
                try:
                    values.append(float(p))
                except ValueError:
                    pass
            if values:
                result["methods"][name] = values

    return result


def write_regret_csv(parsed: dict, horizons: list[int], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"T={t}" for t in horizons])
        for name, values in parsed["methods"].items():
            row = [name] + [f"{v:.1f}" for v in values]
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--output", default="results/parsed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(results_dir.rglob("eval_bandit*.txt")):
        text = txt_file.read_text()
        parsed = parse_bandit_eval(text)
        if not parsed["methods"]:
            continue

        horizons_match = re.findall(r"T=\s*(\d+)", text.split("\n")[1] if len(text.split("\n")) > 1 else "")
        if not horizons_match:
            horizons_match = re.findall(r"T=\s*(\d+)", text)
        horizons = [int(h) for h in horizons_match[:len(next(iter(parsed["methods"].values())))]]

        rel = txt_file.relative_to(results_dir)
        csv_name = rel.with_suffix(".csv")
        write_regret_csv(parsed, horizons, output_dir / csv_name)
        print(f"  {txt_file.name} -> {csv_name}")


if __name__ == "__main__":
    main()

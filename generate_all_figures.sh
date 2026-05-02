#!/bin/bash
# Generate all paper figures and tables from results CSVs.
# Run from band_pfn/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== Step 1: Parse eval .txt files into CSVs ==="
python3 scripts/parse_eval.py results/ --output results/parsed/

echo ""
echo "=== Step 2: Generate regret curve plots ==="
mkdir -p figures

# K=3 mixed (main result)
python3 scripts/plot_regret_curves.py \
    --csvs results/parsed/k3_mixed/eval_bandit_mixed_k3d3_warmstart.csv \
    --title "K=3, Mixed Priors" \
    --output figures/regret_k3_mixed.pdf

# K=3 formula
python3 scripts/plot_regret_curves.py \
    --csvs results/parsed/k3_formula/eval_bandit_k3d3.csv \
    --title "K=3, Formula Only" \
    --output figures/regret_k3_formula.pdf

# K=5 formula warm-start
python3 scripts/plot_regret_curves.py \
    --csvs results/parsed/k5_formula/eval_bandit_k5d5_warmstart.csv \
    --title "K=5, Formula, Warm-Start" \
    --output figures/regret_k5_formula_ws.pdf

echo ""
echo "=== Step 3: Generate scaling plot ==="
python3 scripts/plot_scaling.py --output figures/scaling.pdf

echo ""
echo "=== Done ==="
ls -la figures/*.pdf

# BANDHA

Reference code and result tables for **BANDHA: Learned Warmup and
Target-Conditioned Handoff for Contextual Bandits**.

BANDHA trains an amortized warmup policy for repeated contextual-bandit
instances. The implementation in this repository is the finite, discrete-arm
tabular instantiation used in the paper: synthetic PFN-style tasks produce
all-arm reward targets after each logged bandit episode, while the model input
contains only the censored online history.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 demo.py
```

For a short training smoke test:

```bash
python3 train.py --smoke
```

For evaluation from a checkpoint:

```bash
python3 eval.py --checkpoint path/to/checkpoint.pt --mode bandit --T 200
```

Large trained checkpoints are not stored in this repository. The included CSVs
record the result summaries used by the paper.

## Anonymous Review Snapshot

An anonymized snapshot for review is available at:

https://anonymous.4open.science/r/bandha-4D3D/

## Contents

| Path | Description |
| --- | --- |
| `envs.py` | Synthetic tabular contextual-bandit task generation |
| `model.py` | Sequence baseline model |
| `model_perarm.py` | Permutation-equivariant per-arm state model |
| `losses.py` | Soft-feedback loss and online bandit evaluation helpers |
| `train.py` | Standalone training loop |
| `eval.py` | Synthetic evaluation from a saved checkpoint |
| `eval_realworld.py` | OpenML classification-to-bandit evaluation |
| `results/submission_20260901/` | Frozen synthetic and handoff result summaries |
| `results/realworld_perarm_pca3/` | Real-world classification transfer summaries |

## Scope

The code here is intentionally standalone and public-facing. Internal
orchestration, cluster runners, process logs, private paths, and model
checkpoints are excluded. The general BANDHA recipe can use other task
generators or reward spaces, but this repository focuses on the tabular
probability-vector target tested in the paper.

## Citation

```bibtex
@inproceedings{bandha2026,
  title     = {{BANDHA}: Learned Warmup and Target-Conditioned Handoff for Contextual Bandits},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
}
```

## Acknowledgments

Synthetic task generation follows the prior-fitted network line of work,
including PFNs and TabPFN.

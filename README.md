# BANDHA: Bandit Amortized Network for Direct Handoff Acceleration

A causal Transformer pretrained on synthetic bandit histories for contextual bandit warmup.

## Quick Start

```bash
pip install torch numpy
python demo.py
```

## What is BANDHA?

Standard contextual bandit algorithms (LinUCB, Thompson Sampling) explore from scratch
in every new environment. BANDHA pretrains a causal Transformer on diverse synthetic
bandit histories, learning to predict the correct arm from partial feedback. At deployment,
it handles the critical first ~200 rounds where uninformed exploration wastes the most budget,
then hands off to a conventional algorithm for asymptotic optimality.

### Key ideas

1. **Synthetic pretraining with counterfactual access.** During training, the full reward
   table is known, enabling dense supervision on all arms — not just the one that was pulled.

2. **Soft arm targets.** Gaussian CDF binning of a latent regression function provides
   smooth, geometry-preserving supervision.

3. **Teacher-forced training.** One parallel forward pass with a causal mask, no sequential
   rollout during training. The model sees logged random histories and predicts the correct
   arm at every position.

4. **Greedy deployment.** At inference, the model selects arms by argmax. No Thompson Sampling
   or UCB — the amortized prior is strong enough that exploitation beats uninformed exploration.

## Files

| File | Description |
|------|-------------|
| `envs.py` | Environment generation from PFNs-style synthetic priors |
| `model.py` | Causal Transformer architecture |
| `losses.py` | Training loss (soft CE) and bandit evaluation |
| `train.py` | Training loop |
| `eval.py` | Evaluation (offline + online bandit regret) |
| `demo.py` | End-to-end demo |

## Citation

```bibtex
@inproceedings{bandha2027,
  title     = {{BANDHA}: Bandit Amortized Network for Direct Handoff Acceleration},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year      = {2027},
}
```

## Acknowledgments

Environment generation adapted from the [PFNs](https://github.com/SamuelGabriel/PFNs)
synthetic prior framework (Apache-2.0).

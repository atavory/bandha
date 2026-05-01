"""Bandit environments via PFNs synthetic priors.

Each environment is a K-armed contextual bandit derived from a synthetic
regression problem. A random latent function f maps contexts to continuous
scores; these are converted to soft arm probabilities via Gaussian CDF
soft binning around calibration-derived thresholds.

At interaction time, the true arm is sampled from the soft distribution,
and rewards are binary (1 if correct, 0 otherwise). But the environment
also exposes arm_means[t, k] = P(arm k is correct | x_t) for dense
supervision during training.

Data generation follows the Prior-Fitted Networks (PFNs) approach:
  Muller et al., "Transformers Can Do Bayesian Inference", ICLR 2022
  Hollmann et al., "TabPFN", ICLR 2023
  Source: github.com/SamuelGabriel/PFNs (Apache-2.0)
Adapted from fbcode/pytorch/PFNs/pfns/priors/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Latent function samplers (PFNs prior types)
# ---------------------------------------------------------------------------

# --- Formula tree prior ---

BINARY_OPS: dict[str, Callable] = {
    "add": lambda x, y: x + y,
    "mul": lambda x, y: x * y,
    "gate": lambda x, y: (x > 0).float() * y,
    "power": lambda x, y: x.abs() ** y.clamp(-2, 2),
}

UNARY_OPS: dict[str, Callable] = {
    "abs": lambda x: x.abs(),
    "sin": lambda x: torch.sin(x * 4 * math.pi),
    "exp": lambda x: torch.exp(x.clamp(-5, 5)),
    "log": lambda x: torch.log(x.abs() + 1e-8),
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "square": lambda x: x ** 2,
    "sigmoid": lambda x: torch.sigmoid(x),
    "relu": lambda x: torch.relu(x),
}

@dataclass
class Leaf:
    input_idx: int

@dataclass
class Unary:
    op: str
    child: Leaf | Unary | Binary
    factor: float = 1.0
    bias: float = 0.0

@dataclass
class Binary:
    op: str
    left: Leaf | Unary | Binary
    right: Leaf | Unary | Binary
    factor_left: float = 1.0
    factor_right: float = 1.0
    bias_left: float = 0.0
    bias_right: float = 0.0

Node = Leaf | Unary | Binary


def _bound(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _eval_tree(node: Node, inputs: torch.Tensor) -> torch.Tensor:
    if isinstance(node, Leaf):
        return inputs[:, node.input_idx]
    if isinstance(node, Unary):
        child_val = _eval_tree(node.child, inputs)
        return _bound(UNARY_OPS[node.op](node.factor * child_val + node.bias))
    if isinstance(node, Binary):
        left_val = _eval_tree(node.left, inputs)
        right_val = _eval_tree(node.right, inputs)
        left_t = node.factor_left * left_val + node.bias_left
        right_t = node.factor_right * right_val + node.bias_right
        return _bound(BINARY_OPS[node.op](left_t, right_t))
    raise TypeError(f"Unknown node type: {type(node)}")


def _sample_tree(d: int, rng: np.random.Generator, num_leaves: int = 4) -> Node:
    binary_ops = list(BINARY_OPS.keys())
    unary_ops = list(UNARY_OPS.keys())
    leaves: list[Leaf] = [Leaf(rng.integers(d)) for _ in range(num_leaves)]

    def maybe_wrap(node: Node) -> Node:
        if rng.random() < 0.5:
            op = unary_ops[rng.integers(len(unary_ops))]
            return Unary(op, node, rng.standard_normal() * 0.5 + 1.0, rng.standard_normal() * 0.3)
        return node

    nodes = [maybe_wrap(leaf) for leaf in leaves]
    while len(nodes) > 1:
        rng.shuffle(nodes)
        left, right = nodes.pop(), nodes.pop()
        op = binary_ops[rng.integers(len(binary_ops))]
        node = Binary(op, left, right,
                      rng.standard_normal() * 0.5 + 1.0, rng.standard_normal() * 0.5 + 1.0,
                      rng.standard_normal() * 0.3, rng.standard_normal() * 0.3)
        nodes.append(maybe_wrap(node))
    return nodes[0]


def sample_formula_fn(d: int, rng: np.random.Generator) -> Callable:
    tree = _sample_tree(d, rng)
    return lambda x: _eval_tree(tree, x)


# --- Random MLP prior ---

def sample_mlp_fn(d: int, rng: np.random.Generator, n_layers: int = 2, n_hidden: int = 64) -> Callable:
    seed = int(rng.integers(0, 2**31))
    gen = torch.Generator().manual_seed(seed)
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    d_in = d
    for i in range(n_layers - 1):
        w = torch.randn(n_hidden, d_in, generator=gen) * 0.1
        b = torch.randn(n_hidden, generator=gen) * 0.1
        layers.append((w, b))
        d_in = n_hidden
    w_out = torch.randn(1, d_in, generator=gen) * 0.1
    b_out = torch.randn(1, generator=gen) * 0.1
    layers.append((w_out, b_out))

    def fn(x: torch.Tensor) -> torch.Tensor:
        h = x
        for w, b in layers[:-1]:
            h = torch.tanh(h @ w.T + b)
        w, b = layers[-1]
        return (h @ w.T + b).squeeze(-1)

    return fn


# --- Random GP prior (RBF kernel via random Fourier features) ---

def sample_gp_fn(d: int, rng: np.random.Generator, n_basis: int = 50) -> Callable:
    seed = int(rng.integers(0, 2**31))
    gen = torch.Generator().manual_seed(seed)
    length_scale = float(rng.uniform(0.5, 2.0))
    W = torch.randn(n_basis, d, generator=gen) / length_scale
    b = torch.rand(n_basis, generator=gen) * 2 * math.pi
    alpha = torch.randn(n_basis, generator=gen)

    def fn(x: torch.Tensor) -> torch.Tensor:
        proj = x @ W.T + b
        features = torch.cos(proj) * math.sqrt(2.0 / n_basis)
        return features @ alpha

    return fn


# ---------------------------------------------------------------------------
# Prior bag
# ---------------------------------------------------------------------------

def sample_latent_fn(d: int, rng: np.random.Generator, prior_type: str = "mixed") -> Callable:
    if prior_type == "mixed":
        prior_type = rng.choice(["formula", "mlp", "gp"])
    if prior_type == "formula":
        return sample_formula_fn(d, rng)
    elif prior_type == "mlp":
        return sample_mlp_fn(d, rng)
    elif prior_type == "gp":
        return sample_gp_fn(d, rng)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")


# ---------------------------------------------------------------------------
# Soft binning: Gaussian CDF over threshold intervals
# ---------------------------------------------------------------------------

_NORMAL = torch.distributions.Normal(0, 1)

# Fixed multiplier for environment-specific boundary softness.
# The actual sigma is sigma_mult * median calibration bin width.
SOFT_BINNING_SIGMA_MULT = 0.25


def soft_arm_probs(
    scores: torch.Tensor,
    thresholds: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Convert normalized latent scores to soft arm probabilities.

    P(arm=k | m) = Φ((τ_k - m) / σ) - Φ((τ_{k-1} - m) / σ)

    Args:
        scores:     [n] latent scores
        thresholds: [K-1] sorted thresholds
        sigma:      smoothing bandwidth

    Returns:
        probs: [n, K] soft arm probabilities, sum to 1 per row
    """
    # extended thresholds: [-inf, τ_1, ..., τ_{K-1}, +inf]
    K = len(thresholds) + 1
    ext = torch.cat([
        torch.tensor([-1e6], dtype=scores.dtype, device=scores.device),
        thresholds.to(dtype=scores.dtype, device=scores.device),
        torch.tensor([1e6], dtype=scores.dtype, device=scores.device),
    ])

    # Φ((τ_k - m) / σ) for all k and all m
    m = scores.unsqueeze(-1)  # [n, 1]
    cdf_vals = _NORMAL.cdf((ext - m) / sigma)  # [n, K+1]

    # P(arm=k) = Φ(τ_{k+1}) - Φ(τ_k)
    probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]  # [n, K]

    # clamp for numerical safety
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


def _sigma_from_thresholds(
    thresholds: torch.Tensor,
    calib_scores: torch.Tensor,
    K: int,
    sigma_mult: float,
) -> float:
    if sigma_mult <= 0:
        return 1e-6

    if thresholds.numel() > 1:
        widths = thresholds[1:] - thresholds[:-1]
        widths = widths[widths > 1e-6]
        if widths.numel() > 0:
            return max(float(sigma_mult * widths.median().item()), 1e-6)

    fallback = float(calib_scores.std().item() / max(K, 1))
    return max(sigma_mult * fallback, 1e-6)


# ---------------------------------------------------------------------------
# Bandit environment
# ---------------------------------------------------------------------------

@dataclass
class BanditEnv:
    """A K-armed contextual bandit with soft arm probabilities.

    arm_means[i, k] = P(arm k is correct | x_i) — soft targets for training.
    true_arms[i] = sampled correct arm — for bandit interaction.
    """
    contexts: torch.Tensor    # [n, d]
    arm_means: torch.Tensor   # [n, K] — soft arm probabilities
    true_arms: torch.Tensor   # [n] long — sampled correct arm
    K: int

    @property
    def n(self) -> int:
        return self.contexts.shape[0]

    @property
    def d(self) -> int:
        return self.contexts.shape[1]

    def step(self, t: int, arm: int, env_seed: int = 0) -> float:
        return 1.0 if arm == self.true_arms[t].item() else 0.0

    def optimal_arm(self, t: int) -> int:
        return int(self.arm_means[t].argmax().item())

    def regret(self, t: int, arm: int) -> float:
        return float(self.arm_means[t].max() - self.arm_means[t, arm])


def make_env(
    K: int = 5,
    d: int = 10,
    n: int = 500,
    n_calib: int = 500,
    sigma_mult: float = SOFT_BINNING_SIGMA_MULT,
    rng: np.random.Generator | None = None,
    prior_type: str = "formula",
) -> BanditEnv:
    """Generate one bandit environment with soft arm probabilities.

    1. Sample a random latent function f from the prior bag.
    2. Compute calibration-derived thresholds from an independent sample.
    3. Compute soft arm probabilities via Gaussian CDF soft binning.
    4. Sample true arms from the soft distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    fn = sample_latent_fn(d, rng, prior_type=prior_type)

    # calibration: thresholds from an independent sample
    calib_seed = int(rng.integers(0, 2**31))
    calib_ctx = torch.randn(n_calib, d, generator=torch.Generator().manual_seed(calib_seed))
    calib_y = fn(calib_ctx)
    quantiles = torch.linspace(0, 1, K + 1)[1:-1]
    thresholds = torch.quantile(calib_y.float(), quantiles)
    sigma = _sigma_from_thresholds(thresholds, calib_y.float(), K, sigma_mult)

    # episode contexts
    ctx_seed = int(rng.integers(0, 2**31))
    contexts = torch.randn(n, d, generator=torch.Generator().manual_seed(ctx_seed))
    y = fn(contexts)

    # soft arm probabilities
    arm_means = soft_arm_probs(y.float(), thresholds.float(), sigma)

    # permute arm labels
    perm_seed = int(rng.integers(0, 2**31))
    perm_gen = torch.Generator().manual_seed(perm_seed)
    perm = torch.randperm(K, generator=perm_gen)
    arm_means = arm_means[:, perm]

    # sample true arms from soft distribution
    sample_seed = int(rng.integers(0, 2**31))
    sample_gen = torch.Generator().manual_seed(sample_seed)
    true_arms = torch.multinomial(arm_means, 1, generator=sample_gen).squeeze(-1)

    return BanditEnv(contexts=contexts, arm_means=arm_means, true_arms=true_arms, K=K)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for i in range(5):
        env = make_env(K=5, d=10, n=200, rng=rng)
        arm_counts = torch.bincount(env.true_arms, minlength=env.K)
        fracs = arm_counts.float() / env.n
        max_prob = env.arm_means.max(dim=-1).values.mean()
        entropy = -(env.arm_means * (env.arm_means + 1e-8).log()).sum(dim=-1).mean()
        print(
            f"Env {i}: d={env.d} K={env.K} n={env.n} "
            f"arm_fracs={[f'{f:.2f}' for f in fracs.tolist()]} "
            f"mean_max_prob={max_prob:.3f} "
            f"mean_entropy={entropy:.3f} "
            f"regret(0,wrong)={env.regret(0, (env.optimal_arm(0) + 1) % env.K):.3f}"
        )

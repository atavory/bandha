"""Multi-worker DataLoader for bandit environment generation.

Generates (contexts, arm_means, true_arms) batches in parallel using
PyTorch DataLoader with multiple worker processes. Each worker has its
own RNG seed for reproducibility.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from envs import make_env
from losses import sample_logged_history, build_reward_history


class BanditEnvDataset(IterableDataset):
    """IterableDataset that generates bandit environments on the fly.

    Each item is (contexts, arm_means, true_arms) for one environment.
    Workers get non-overlapping RNG seeds for reproducibility.
    """

    def __init__(
        self,
        K: int,
        d: int,
        T: int,
        n_envs: int,
        prior_type: str = "formula",
        base_seed: int = 0,
        epoch: int = 0,
    ):
        self.K = K
        self.d = d
        self.T = T
        self.n_envs = n_envs
        self.prior_type = prior_type
        self.base_seed = base_seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        per_worker = self.n_envs // num_workers
        leftover = self.n_envs % num_workers
        my_count = per_worker + (1 if worker_id < leftover else 0)

        seed = self.base_seed + self.epoch * 100000 + worker_id * 10000
        rng = np.random.default_rng(seed)

        for _ in range(my_count):
            env = make_env(K=self.K, d=self.d, n=self.T, rng=rng,
                           prior_type=self.prior_type)
            yield env.contexts, env.arm_means, env.true_arms


class PrecomputedEnvDataset(IterableDataset):
    """Dataset backed by a pre-generated env bank.

    Samples random indices from the bank each epoch. The logged history
    (random arm pulls) is re-sampled each time, so the model sees different
    trajectories even on cached environments.
    """

    def __init__(
        self,
        contexts: torch.Tensor,
        arm_means: torch.Tensor,
        true_arms: torch.Tensor,
        n_envs_per_epoch: int,
        base_seed: int = 0,
        epoch: int = 0,
    ):
        self.contexts = contexts
        self.arm_means = arm_means
        self.true_arms = true_arms
        self.bank_size = contexts.shape[0]
        self.n_envs_per_epoch = n_envs_per_epoch
        self.base_seed = base_seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.base_seed + self.epoch * 100000)
        indices = rng.choice(self.bank_size, size=self.n_envs_per_epoch,
                             replace=self.n_envs_per_epoch > self.bank_size)
        for idx in indices:
            yield self.contexts[idx], self.arm_means[idx], self.true_arms[idx]


def collate_envs(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts, arm_means, true_arms = zip(*batch)
    return torch.stack(contexts), torch.stack(arm_means), torch.stack(true_arms)


def make_train_dataloader(
    K: int,
    d: int,
    T: int,
    n_envs: int,
    batch_size: int,
    prior_type: str = "formula",
    num_workers: int = 0,
    base_seed: int = 0,
    epoch: int = 0,
    env_bank: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for training.

    If env_bank is provided, uses the precomputed bank.
    Otherwise generates environments on the fly with num_workers parallel workers.
    """
    if env_bank is not None:
        dataset = PrecomputedEnvDataset(
            *env_bank,
            n_envs_per_epoch=n_envs,
            base_seed=base_seed,
            epoch=epoch,
        )
    else:
        dataset = BanditEnvDataset(
            K=K, d=d, T=T, n_envs=n_envs,
            prior_type=prior_type,
            base_seed=base_seed,
            epoch=epoch,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_envs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

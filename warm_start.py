from __future__ import annotations

from dataclasses import dataclass

import torch
from model import BanditPFN


@dataclass
class WarmStartSummary:
    checkpoint_path: str
    source_k: int
    source_d_ctx: int
    copied_keys: list[str]
    skipped_keys: list[str]


def _copy_matching_tensors(
    model: BanditPFN,
    source_state: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    skipped: list[str] = []
    target_state = model.state_dict()
    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None or source_value.shape != target_value.shape:
            skipped.append(key)
            continue
        target_value.copy_(source_value)
        copied.append(key)
    return copied, skipped


def _copy_input_projection(
    model: BanditPFN,
    source_state: dict[str, torch.Tensor],
    source_d_ctx: int,
    source_k: int,
) -> list[str]:
    copied: list[str] = []
    weight = source_state.get("input_proj.weight")
    if weight is not None and weight.shape[0] == model.input_proj.weight.shape[0]:
        dst = model.input_proj.weight.data
        ctx_cols = min(source_d_ctx, model.d_ctx)
        if ctx_cols > 0:
            dst[:, :ctx_cols] = weight[:, :ctx_cols]
        src_reward = weight[:, source_d_ctx : source_d_ctx + source_k]
        dst_reward = dst[:, model.d_ctx : model.d_ctx + model.K]
        shared = min(source_k, model.K)
        if shared > 0:
            dst_reward[:, :shared] = src_reward[:, :shared]
        if model.K > shared and source_k > 0:
            dst_reward[:, shared:] = src_reward.mean(dim=1, keepdim=True)
        copied.append("input_proj.weight[partial]")
    return copied


def load_warm_start(
    model: BanditPFN,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> WarmStartSummary:
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    source_args = ckpt["args"]
    source_state = ckpt["model_state_dict"]
    copied, skipped = _copy_matching_tensors(model, source_state)

    source_d_ctx = int(source_args["d"])
    source_k = int(source_args["K"])
    if "input_proj.weight" in skipped:
        partial_copied = _copy_input_projection(
            model,
            source_state,
            source_d_ctx=source_d_ctx,
            source_k=source_k,
        )
        if partial_copied:
            copied.extend(partial_copied)
            skipped = [key for key in skipped if key != "input_proj.weight"]

    return WarmStartSummary(
        checkpoint_path=checkpoint_path,
        source_k=source_k,
        source_d_ctx=source_d_ctx,
        copied_keys=sorted(set(copied)),
        skipped_keys=sorted(skipped),
    )

"""BanditPFN training — teacher-forced on logged bandit histories.

Pure PyTorch, no TorchTNT. Generates random trajectories, one forward
pass with causal mask, CE on soft arm targets.

Usage:
    python3 -m band_pfn.algs.train --smoke
    python3 -m band_pfn.algs.train --K 3 --d 3 --T 200 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from envs import make_env
from losses import loss_fn
from model import BanditPFN
from warm_start import load_warm_start

CHECKPOINT_STRUCTURAL_KEYS = (
    "K",
    "T",
    "d",
    "d_model",
    "n_heads",
    "n_layers",
    "ff_mult",
    "max_T",
    "prior_type",
)


def make_batch(
    n_envs: int,
    K: int,
    d: int,
    T: int,
    rng: np.random.Generator,
    prior_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctxs, ams, arms = [], [], []
    for _ in range(n_envs):
        env = make_env(K=K, d=d, n=T, rng=rng, prior_type=prior_type)
        ctxs.append(env.contexts)
        ams.append(env.arm_means)
        arms.append(env.true_arms)
    return torch.stack(ctxs), torch.stack(ams), torch.stack(arms)


def make_env_bank(
    n_envs: int,
    K: int,
    d: int,
    T: int,
    rng: np.random.Generator,
    prior_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return make_batch(n_envs, K, d, T, rng, prior_type)


def sample_bank_indices(
    bank_size: int,
    n_envs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    replace = n_envs > bank_size
    return rng.choice(bank_size, size=n_envs, replace=replace)


def make_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_checkpoint_stem(args: argparse.Namespace) -> str:
    return f"bandit_pfn_{args.tag}_seed{args.seed}"


def truncate_log_to_epoch(log_path: Path, max_epoch: int) -> None:
    if not log_path.exists():
        return
    with log_path.open(newline="") as infile:
        rows = list(csv.reader(infile))
    if not rows:
        return
    header, data_rows = rows[0], rows[1:]
    kept_rows = []
    for row in data_rows:
        if not row:
            continue
        try:
            epoch = int(row[0])
        except ValueError:
            continue
        if epoch <= max_epoch:
            kept_rows.append(row)
    with log_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(kept_rows)


def save_checkpoint(
    path: Path,
    *,
    model: BanditPFN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
    epoch: int,
    best_val_norm: float,
    rng: np.random.Generator,
    env_bank: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "args": vars(args),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_norm": best_val_norm,
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "env_bank": None if env_bank is None else tuple(t.cpu() for t in env_bank),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def validate_resume_args(
    saved_args: dict[str, object],
    current_args: argparse.Namespace,
) -> None:
    mismatches = []
    for key in CHECKPOINT_STRUCTURAL_KEYS:
        saved_value = saved_args.get(key)
        current_value = getattr(current_args, key)
        if saved_value != current_value:
            mismatches.append((key, saved_value, current_value))
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r} current={current!r}"
            for key, saved, current in mismatches
        )
        raise ValueError(f"Resume checkpoint args mismatch: {details}")


def run_validation(
    model: BanditPFN,
    *,
    n_envs: int,
    K: int,
    d: int,
    T: int,
    device: torch.device,
    seed: int,
    prior_type: str,
    loss_start_frac: float,
    precision: str,
) -> tuple[float, float, float]:
    model.eval()
    val_rng = np.random.default_rng(seed)
    va_list, vf_list, vn_list = [], [], []
    for _ in range(n_envs):
        env = make_env(K=K, d=d, n=T, rng=val_rng, prior_type=prior_type)
        with torch.no_grad():
            with make_autocast_context(device, precision):
                result = loss_fn(
                    model,
                    env.contexts.unsqueeze(0).to(device),
                    env.arm_means.unsqueeze(0).to(device),
                    env.true_arms.unsqueeze(0).to(device),
                    loss_start_frac=loss_start_frac,
                )
        va_list.append(result.metrics["accuracy"])
        vf_list.append(result.metrics["accuracy_final20"])
        vn_list.append(result.metrics["regret_norm"])
    return (
        float(np.mean(va_list)),
        float(np.mean(vf_list)),
        float(np.mean(vn_list)),
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_model(args: argparse.Namespace, device: torch.device) -> BanditPFN:
    return BanditPFN(
        d_ctx=args.d,
        K=args.K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_T=args.max_T,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--max-T", type=int, default=1024)
    parser.add_argument("--prior-type", type=str, default="formula")
    parser.add_argument("--n-envs", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--env-bank-size", type=int, default=0)
    parser.add_argument("--env-bank-refresh", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss-start-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tag", type=str, default="v9")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    args = parser.parse_args(argv)

    if args.smoke:
        args.n_envs = 32
        args.n_val = 8
        args.T = 50
        args.max_T = 128
        args.epochs = 3
        args.batch_size = 8
        args.d_model = 64
        args.n_layers = 2
        args.tag = "v9_smoke"

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    if not 0.0 <= args.loss_start_frac < 1.0:
        raise ValueError("--loss-start-frac must be in [0, 1)")
    if args.resume_from and args.warm_start_checkpoint:
        raise ValueError(
            "Use either --resume-from or --warm-start-checkpoint, not both"
        )
    args.max_T = max(args.max_T, args.T)

    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(
            args.resume_from, map_location="cpu", weights_only=False
        )
        args.max_T = int(resume_checkpoint["args"].get("max_T", args.max_T))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" and args.precision != "fp32":
        raise ValueError("Mixed precision is only supported on CUDA")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = build_model(args, device)
    if args.warm_start_checkpoint:
        summary = load_warm_start(
            model, args.warm_start_checkpoint, map_location=device
        )
        print(
            f"Warm start: source K={summary.source_k} d={summary.source_d_ctx} "
            f"copied={len(summary.copied_keys)} skipped={len(summary.skipped_keys)}"
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BanditPFN v9: {n_params:,} params | device={device}")
    print(f"K={args.K} d={args.d} T={args.T} prior={args.prior_type}")
    print(f"n_envs={args.n_envs} epochs={args.epochs} batch={args.batch_size}")
    print(f"Random baseline: acc={1 / args.K:.1%}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )

    checkpoint_stem = make_checkpoint_stem(args)
    log_path = save_dir / f"train_log_{args.tag}_seed{args.seed}.csv"
    final_path = save_dir / f"{checkpoint_stem}.pt"
    best_path = save_dir / f"{checkpoint_stem}_best.pt"

    start_epoch = 1
    resume_epoch = 0
    best_val_norm = float("inf")
    env_bank = None

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        validate_resume_args(checkpoint["args"], args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_norm = float(checkpoint.get("best_val_norm", float("inf")))
        start_epoch = int(checkpoint["epoch"]) + 1
        rng.bit_generator.state = checkpoint["rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        loaded_bank = checkpoint.get("env_bank")
        if loaded_bank is not None:
            env_bank = tuple(t.cpu() for t in loaded_bank)
        resume_epoch = int(checkpoint["epoch"])
        print(
            f"Resumed {args.resume_from} from epoch {checkpoint['epoch']} "
            f"(best_val_norm={best_val_norm:.3f})"
        )

    if resume_epoch > 0:
        truncate_log_to_epoch(log_path, resume_epoch)

    log_mode = "a" if start_epoch > 1 and log_path.exists() else "w"
    log_file = open(log_path, log_mode, newline="")
    writer = csv.writer(log_file)
    if log_mode == "w":
        writer.writerow(
            [
                "epoch",
                "loss",
                "acc",
                "acc_final20",
                "norm",
                "early",
                "late",
                "val_acc",
                "val_acc_f20",
                "val_norm",
                "time_s",
            ]
        )

    print(
        f"\n{'Ep':>3s} {'Loss':>7s} {'Acc':>6s} {'F20':>6s} {'Norm':>6s} "
        f"{'VAcc':>6s} {'VF20':>6s} {'VNorm':>6s} {'T':>5s}"
    )
    print("-" * 58)

    if env_bank is None and args.env_bank_size > 0:
        t0 = time.time()
        env_bank = make_env_bank(
            args.env_bank_size,
            args.K,
            args.d,
            args.T,
            rng,
            args.prior_type,
        )
        print(
            f"Pre-generated env bank: size={args.env_bank_size} "
            f"refresh={args.env_bank_refresh or 'never'} "
            f"({time.time() - t0:.1f}s)"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss, ep_acc, ep_f20, ep_norm, ep_early, ep_late = [], [], [], [], [], []

        if (
            args.env_bank_size > 0
            and args.env_bank_refresh > 0
            and epoch > 1
            and (epoch - 1) % args.env_bank_refresh == 0
        ):
            env_bank = make_env_bank(
                args.env_bank_size,
                args.K,
                args.d,
                args.T,
                rng,
                args.prior_type,
            )

        active_bank_size = args.env_bank_size
        if env_bank is not None:
            active_bank_size = int(env_bank[0].shape[0])

        bank_indices = None
        if env_bank is not None:
            bank_indices = sample_bank_indices(active_bank_size, args.n_envs, rng)

        optimizer.zero_grad(set_to_none=True)
        micro_batches_since_step = 0

        for i in range(0, args.n_envs, args.batch_size):
            bs = min(args.batch_size, args.n_envs - i)
            if env_bank is None:
                ctx, am, arms = make_batch(
                    bs,
                    args.K,
                    args.d,
                    args.T,
                    rng,
                    args.prior_type,
                )
            else:
                idx = torch.from_numpy(bank_indices[i : i + bs]).long()
                ctx = env_bank[0][idx]
                am = env_bank[1][idx]
                arms = env_bank[2][idx]

            ctx = ctx.to(device)
            am = am.to(device)
            arms = arms.to(device)

            with make_autocast_context(device, args.precision):
                result = loss_fn(
                    model,
                    ctx,
                    am,
                    arms,
                    loss_start_frac=args.loss_start_frac,
                )
                micro_loss = result.loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()

            micro_batches_since_step += 1
            should_step = (
                micro_batches_since_step == args.grad_accum_steps
                or i + bs >= args.n_envs
            )

            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_batches_since_step = 0

            ep_loss.append(result.loss.item())
            ep_acc.append(result.metrics["accuracy"])
            ep_f20.append(result.metrics["accuracy_final20"])
            ep_norm.append(result.metrics["regret_norm"])
            ep_early.append(result.metrics["regret_early"])
            ep_late.append(result.metrics["regret_late"])

        scheduler.step()

        va, vf, vn = 0.0, 0.0, 0.0
        ran_validation = epoch % args.eval_every == 0 or epoch == args.epochs
        if ran_validation:
            va, vf, vn = run_validation(
                model,
                n_envs=args.n_val,
                K=args.K,
                d=args.d,
                T=args.T,
                device=device,
                seed=args.seed + 1000 + epoch,
                prior_type=args.prior_type,
                loss_start_frac=args.loss_start_frac,
                precision=args.precision,
            )
            if vn < best_val_norm:
                best_val_norm = vn
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if scaler.is_enabled() else None,
                    args=args,
                    epoch=epoch,
                    best_val_norm=best_val_norm,
                    rng=rng,
                    env_bank=env_bank,
                )

        elapsed = time.time() - t0
        ml = float(np.mean(ep_loss))
        ma = float(np.mean(ep_acc))
        mf = float(np.mean(ep_f20))
        mn = float(np.mean(ep_norm))
        me = float(np.mean(ep_early))
        mlate = float(np.mean(ep_late))
        print(
            f"{epoch:3d} {ml:7.4f} {ma:5.1%} {mf:5.1%} {mn:6.3f} "
            f"{va:5.1%} {vf:5.1%} {vn:6.3f} {elapsed:4.0f}s"
        )
        writer.writerow(
            [
                epoch,
                f"{ml:.4f}",
                f"{ma:.4f}",
                f"{mf:.4f}",
                f"{mn:.4f}",
                f"{me:.4f}",
                f"{mlate:.4f}",
                f"{va:.4f}",
                f"{vf:.4f}",
                f"{vn:.4f}",
                f"{elapsed:.1f}",
            ]
        )
        log_file.flush()

        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = save_dir / f"{checkpoint_stem}_ep{epoch:04d}.pt"
            save_checkpoint(
                periodic_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                args=args,
                epoch=epoch,
                best_val_norm=best_val_norm,
                rng=rng,
                env_bank=env_bank,
            )

    log_file.close()
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler if scaler.is_enabled() else None,
        args=args,
        epoch=args.epochs,
        best_val_norm=best_val_norm,
        rng=rng,
        env_bank=env_bank,
    )
    print(f"\nSaved {final_path} | log: {log_path}")
    if best_val_norm < float("inf"):
        print(f"Best checkpoint: {best_path} | best val norm: {best_val_norm:.3f}")


if __name__ == "__main__":
    main()

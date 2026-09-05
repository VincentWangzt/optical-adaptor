from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, set_seed
from safetensors.torch import save_file
from transformers import get_constant_schedule_with_warmup

from optical_adaptor.training.cache import TensorCache
from optical_adaptor.training.config import (
    AdapterKind,
    Pipeline,
    fingerprint,
    load_credentials,
    load_pipeline,
    write_json,
)
from optical_adaptor.training.data import load_manifest
from optical_adaptor.training.models import FrozenQwen, build_adapter
from optical_adaptor.training.objectives import mean_metrics, task_loss
from optical_adaptor.training.sampling import TaskWindow, epoch_windows


def run_update(
    accelerator: Accelerator,
    adapter,
    qwen,
    cache,
    optimizer,
    window: TaskWindow,
    *,
    microbatch_size: int,
    chunk_size: int,
    max_grad_norm: float,
) -> tuple[torch.Tensor, float]:
    """One global pair update, with token normalization before DDP's rank averaging."""
    optimizer.zero_grad(set_to_none=True)
    statistics = torch.zeros(2, 6, device=accelerator.device, dtype=torch.float64)
    for task_index, task in enumerate(("continuation", "reconstruction")):
        rows = getattr(window, task)
        denominator = getattr(window, f"{task}_tokens")
        for start in range(0, len(rows), microbatch_size):
            batch = rows[start : start + microbatch_size]
            last = task == "reconstruction" and start + microbatch_size >= len(rows)
            context = contextlib.nullcontext() if last else accelerator.no_sync(adapter)
            with context, accelerator.autocast():
                visual = cache.batch(batch, "encoder", accelerator.device)
                adapted = adapter(visual).to(visual.dtype)
                hidden, targets, text_mask = qwen.student_hidden(batch, adapted, task)
                teacher = (
                    cache.batch(batch, "teacher", accelerator.device) if task_index == 0 else None
                )
                result = task_loss(
                    hidden,
                    targets,
                    text_mask,
                    head=qwen.model.lm_head,
                    teacher=teacher,
                    chunk_size=chunk_size,
                )
                # Accelerate accumulation is deliberately 1; we own complete update windows.
                scaled = result.loss * (0.5 * accelerator.num_processes / denominator)
                accelerator.backward(scaled)
                statistics[task_index] += result.statistics
            del adapted, hidden, result, scaled
    norm = accelerator.clip_grad_norm_(adapter.parameters(), max_grad_norm)
    if not torch.isfinite(norm):
        raise RuntimeError("non-finite adapter gradient norm")
    optimizer.step()
    return accelerator.reduce(statistics, reduction="sum"), float(norm)


def runtime_identity(pipeline: Pipeline, kind: str, world_size: int) -> dict:
    import accelerate
    import transformers

    return {
        "config": pipeline.config.model_dump(),
        "data": pipeline.data_fingerprint,
        "kind": kind,
        "world_size": world_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "fla_core": version("fla-core"),
        "flash_linear_attention": version("flash-linear-attention"),
        "triton": version("triton"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
    }


def profile_path(pipeline: Pipeline, kind: str, microbatch: int) -> Path:
    return pipeline.output / "profiles" / kind / f"microbatch-{microbatch}.json"


def choose_microbatch(pipeline: Pipeline, kind: str, world_size: int) -> int:
    identity = fingerprint(runtime_identity(pipeline, kind, world_size))
    results = []
    for candidate in pipeline.config.training.microbatch_candidates:
        path = profile_path(pipeline, kind, candidate)
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != identity:
                raise ValueError("profile configuration/runtime mismatch")
            if result["passed"]:
                results.append(result)
    if not results:
        raise ValueError(f"no passing throughput profile for {kind}; run --mode profile first")
    return min(results, key=lambda row: (row["seconds_per_update"], row["microbatch"]))[
        "microbatch"
    ]


def save_checkpoint(
    pipeline: Pipeline,
    accelerator: Accelerator,
    adapter,
    directory: Path,
    metadata: dict,
) -> None:
    # All ranks save their RNG state. Only the prepared adapter and optimizer are registered.
    temporary = directory.with_name(directory.name + ".incomplete")
    accelerator.save_state(str(temporary), safe_serialization=True)
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(adapter)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
            str(temporary / "adapter.safetensors"),
        )
        write_json(
            temporary / "adapter.json",
            {"kind": metadata["kind"], "config": pipeline.config.adapter.model_dump()},
        )
        write_json(temporary / "state.json", metadata)
        if directory.exists():
            raise FileExistsError(f"checkpoint already exists: {directory}")
        temporary.rename(directory)
        periodic = sorted(directory.parent.glob("step-*"))
        for old in periodic[: -pipeline.config.training.keep_periodic]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()


def open_wandb(pipeline: Pipeline, kind: str, mode: str, config: dict, run_id: str | None):
    import wandb

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("this pipeline requires online W&B logging")
    wandb.login(key=os.environ["WANDB_API_KEY"], verify=True)
    return wandb.init(
        entity=pipeline.config.logging.entity,
        project=pipeline.config.logging.project,
        name=f"{kind}-{mode}",
        group="optical-adaptor-v1",
        job_type=mode,
        config=config,
        id=run_id,
        resume="must" if run_id else None,
        mode="online",
        dir=str(pipeline.output),
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )


def train_main(kind: AdapterKind) -> None:
    parser = argparse.ArgumentParser(description=f"Train/profile the {kind} optical adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--mode", choices=("profile", "overfit", "train"), default="train")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-updates", type=int, default=100, help="number of updates for the overfit gate"
    )
    parser.add_argument("--overfit-records", type=int, default=8)
    parser.add_argument(
        "--stop-after", type=int, help="save and exit after this update for resume validation"
    )
    args = parser.parse_args()
    if args.max_updates < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError("update limits must be positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("select idle GPUs explicitly with CUDA_VISIBLE_DEVICES")
    pipeline = load_pipeline(args.config)
    config = pipeline.config
    load_credentials(pipeline, wandb=args.mode != "profile")
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)],
    )
    if (
        accelerator.device.type != "cuda"
        or accelerator.num_processes != 2
        or accelerator.distributed_type != DistributedType.MULTI_GPU
    ):
        raise ValueError("v1 launch requires exactly two CUDA ranks through Accelerate")
    set_seed(config.seed)
    records = load_manifest(pipeline)
    train_records = [row for row in records if row["split"] == "train"]
    identity = runtime_identity(pipeline, kind, accelerator.num_processes)
    microbatch = args.microbatch_size
    if microbatch is None:
        if args.mode == "profile":
            raise ValueError("profile mode requires --microbatch-size")
        microbatch = choose_microbatch(pipeline, kind, accelerator.num_processes)
    if microbatch < 1 or config.training.global_pairs % (microbatch * accelerator.num_processes):
        raise ValueError("microbatch must divide global pairs across ranks")
    cache = TensorCache(pipeline, records)
    qwen = FrozenQwen(pipeline, accelerator.device)
    adapter = build_adapter(kind, config.adapter)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.training.lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
        eps=config.training.epsilon,
    )
    if args.mode == "profile":
        adapter, optimizer = accelerator.prepare(adapter, optimizer)
        longest = sorted(
            train_records, key=lambda row: (-len(row["visual_ids"]), row["record_id"])
        )[: config.training.global_pairs]
        window = next(
            epoch_windows(
                longest,
                seed=config.seed,
                epoch=0,
                global_pairs=config.training.global_pairs,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
            )
        )
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(
            config.training.profile_warmup_updates + config.training.profile_measured_updates
        ):
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_update(
                accelerator,
                adapter,
                qwen,
                cache,
                optimizer,
                window,
                microbatch_size=microbatch,
                chunk_size=config.training.loss_chunk_tokens,
                max_grad_norm=config.training.max_grad_norm,
            )
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - start, device=accelerator.device)
            elapsed = accelerator.gather(elapsed).max().item()
            if accelerator.is_main_process:
                print(
                    f"profile microbatch={microbatch} update={index + 1} seconds={elapsed:.3f}",
                    flush=True,
                )
            if index >= config.training.profile_warmup_updates:
                timings.append(elapsed)
        peak = (
            accelerator.gather(
                torch.tensor(torch.cuda.max_memory_reserved() / 2**30, device=accelerator.device)
            )
            .max()
            .item()
        )
        result = {
            "identity": fingerprint(identity),
            "microbatch": microbatch,
            "seconds_per_update": sum(timings) / len(timings),
            "peak_reserved_gib": peak,
            "passed": peak < config.training.memory_limit_gib,
            "workload": fingerprint([row["record_hash"] for row in longest]),
        }
        if accelerator.is_main_process:
            write_json(profile_path(pipeline, kind, microbatch), result)
            print(json.dumps(result), flush=True)
        accelerator.end_training()
        return

    from optical_adaptor.training.evaluate import evaluate_teacher_forced, generate_reconstructions

    if args.mode == "overfit":
        if (
            args.overfit_records < 2
            or args.overfit_records > len(train_records)
            or args.overfit_records % accelerator.num_processes
        ):
            raise ValueError("overfit record count must be positive and divisible by world size")
        train_records = train_records[: args.overfit_records]
        total_updates = args.max_updates
    else:
        gate_path = pipeline.output / "overfit" / kind / "gate.json"
        gate = json.loads(gate_path.read_text())
        if gate["identity"] != fingerprint(identity) or not gate["passed"]:
            raise ValueError("a passing overfit gate for this configuration is required")
        total_updates = config.total_updates
    updates_per_epoch = math.ceil(len(train_records) / config.training.global_pairs)
    epochs = math.ceil(total_updates / updates_per_epoch)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total_updates * config.training.warmup_ratio)
    )
    adapter, optimizer = accelerator.prepare(adapter, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    run_dir = pipeline.output / ("runs" if args.mode == "train" else "overfit") / kind
    run_dir.mkdir(parents=True, exist_ok=True)
    identity.update(
        mode=args.mode,
        microbatch=microbatch,
        total_updates=total_updates,
        train_records=fingerprint([row["record_hash"] for row in train_records]),
    )
    identity_hash = fingerprint(identity)
    step, start_epoch, start_update, wandb_id = 0, 0, 0, None
    initial = None
    if args.resume:
        metadata = json.loads((args.resume / "state.json").read_text())
        if metadata["identity"] != identity_hash:
            raise ValueError("checkpoint data/configuration/runtime/topology mismatch")
        accelerator.load_state(str(args.resume))
        step, start_epoch, start_update = metadata["step"], metadata["epoch"], metadata["update"]
        wandb_id, initial = metadata["wandb_id"], metadata["initial_metrics"]
    elif (run_dir / "run.json").exists():
        raise FileExistsError(f"run exists; use --resume explicitly: {run_dir}")
    run = (
        open_wandb(pipeline, kind, args.mode, identity, wandb_id)
        if accelerator.is_main_process
        else None
    )
    from accelerate.utils import broadcast_object_list

    wandb_id = broadcast_object_list([run.id if run else None])[0]
    if accelerator.is_main_process:
        write_json(
            run_dir / "run.json",
            {"identity": identity_hash, "resolved": identity, "wandb_id": wandb_id},
        )
    evaluation_records = train_records if args.mode == "overfit" else records
    if initial is None:
        initial = evaluate_teacher_forced(
            pipeline,
            qwen,
            adapter,
            cache,
            evaluation_records,
            accelerator,
            overfit=args.mode == "overfit",
        )
        if run:
            run.log({f"evaluation/{key}": value for key, value in initial.items()}, step=step)
        if accelerator.is_main_process:
            write_json(run_dir / "initial-metrics.json", initial)
    for epoch in range(start_epoch, epochs):
        cursor = start_update if epoch == start_epoch else 0
        windows = epoch_windows(
            train_records,
            seed=config.seed,
            epoch=epoch,
            global_pairs=config.training.global_pairs,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            start_update=cursor,
        )
        for update, window in enumerate(windows, start=cursor):
            adapter.train()
            start = time.perf_counter()
            used_lr = scheduler.get_last_lr()[0]
            statistics, norm = run_update(
                accelerator,
                adapter,
                qwen,
                cache,
                optimizer,
                window,
                microbatch_size=microbatch,
                chunk_size=config.training.loss_chunk_tokens,
                max_grad_norm=config.training.max_grad_norm,
            )
            scheduler.step()
            step += 1
            metrics = {
                f"train/{task}/{key}": value
                for i, task in enumerate(("continuation", "reconstruction"))
                for key, value in mean_metrics(statistics[i], continuation=i == 0).items()
            }
            metrics.update(
                {
                    "train/lr": used_lr,
                    "train/gradient_norm": norm,
                    "train/pairs": window.global_pairs,
                    "system/seconds_per_update": time.perf_counter() - start,
                    "system/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            metrics["train/reconstruction/objective"] = float(
                statistics[1, 0] / window.reconstruction_tokens
            )
            metrics["system/pairs_per_second"] = (
                window.global_pairs / metrics["system/seconds_per_update"]
            )
            if run:
                run.log(metrics, step=step)
            if accelerator.is_main_process:
                print(json.dumps({"step": step, **metrics}), flush=True)
            final = step == total_updates
            if step % config.evaluation.every == 0 or final:
                evaluation = evaluate_teacher_forced(
                    pipeline,
                    qwen,
                    adapter,
                    cache,
                    evaluation_records,
                    accelerator,
                    overfit=args.mode == "overfit",
                )
                if run:
                    run.log(
                        {f"evaluation/{key}": value for key, value in evaluation.items()}, step=step
                    )
                if accelerator.is_main_process:
                    write_json(run_dir / f"evaluation-{step:06d}.json", evaluation)
            if args.mode == "train" and (step % config.evaluation.generation_every == 0 or final):
                generation = generate_reconstructions(
                    pipeline, qwen, adapter, cache, records, accelerator, run_dir, step, final=final
                )
                if run:
                    run.log(
                        {f"generation/{key}": value for key, value in generation.items()}, step=step
                    )
            next_epoch, next_update = epoch, update + 1
            if next_update == updates_per_epoch:
                next_epoch, next_update = epoch + 1, 0
            metadata = {
                "identity": identity_hash,
                "kind": kind,
                "step": step,
                "epoch": next_epoch,
                "update": next_update,
                "wandb_id": wandb_id,
                "initial_metrics": initial,
                "resolved": identity,
            }
            stopping = args.stop_after is not None and step >= args.stop_after
            if final or step % config.training.checkpoint_every == 0 or stopping:
                checkpoint_path = run_dir / ("final" if final else f"step-{step:06d}")
                save_checkpoint(pipeline, accelerator, adapter, checkpoint_path, metadata)
            if final:
                if args.mode == "overfit" and accelerator.is_main_process:
                    passed = (
                        evaluation["continuation/kl"] < initial["continuation/kl"]
                        and evaluation["reconstruction/objective"]
                        < initial["reconstruction/objective"]
                    )
                    write_json(
                        run_dir / "gate.json",
                        {
                            "identity": fingerprint(
                                runtime_identity(pipeline, kind, accelerator.num_processes)
                            ),
                            "passed": passed,
                            "initial": initial,
                            "final": evaluation,
                        },
                    )
                    if not passed:
                        raise RuntimeError("overfit gate failed: both objectives must decrease")
                if run:
                    run.finish()
                accelerator.end_training()
                return
            if stopping:
                if run:
                    run.finish()
                accelerator.end_training()
                return

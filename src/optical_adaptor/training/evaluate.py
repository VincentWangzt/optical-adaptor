from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from safetensors.torch import load_file

from optical_adaptor.edit_distance import evaluate_edit_distance
from optical_adaptor.training.cache import TensorCache
from optical_adaptor.training.config import fingerprint, load_credentials, load_pipeline, write_json
from optical_adaptor.training.data import load_manifest
from optical_adaptor.training.models import FrozenQwen, build_adapter
from optical_adaptor.training.objectives import mean_metrics, task_loss


def evaluation_cases(records: list[dict], *, overfit: bool):
    cases = []
    for record in records:
        split = record["split"]
        if overfit:
            cases.extend((task, record, task) for task in ("continuation", "reconstruction"))
        elif split != "train":
            task = "reconstruction" if split == "reconstruction" else "continuation"
            cases.append((task, record, split))
    return cases


def strata(record: dict) -> list[str]:
    ratio = record["aspect_ratio"]
    aspect = "below-0.75" if ratio < 0.75 else "0.75-to-1.0" if ratio < 1 else "at-least-1.0"
    logical = "40-49" if record["logical_lines"] < 50 else "50-60"
    wrapped = (
        "40-49"
        if record["display_lines"] < 50
        else "50-60"
        if record["display_lines"] <= 60
        else "61-80"
    )
    return [
        "",
        f"language/{record['language']}",
        f"aspect/{aspect}",
        f"logical_lines/{logical}",
        f"display_lines/{wrapped}",
    ]


def aggregate_records(local: dict, accelerator: Accelerator) -> dict[str, float]:
    merged = defaultdict(lambda: torch.zeros(8, dtype=torch.float64))
    for part in gather_object([local]):
        for key, value in part.items():
            merged[key] += torch.tensor(value, dtype=torch.float64)
    metrics = {}
    for key, value in merged.items():
        continuation = key.split("/")[0] != "reconstruction"
        metrics.update(
            {
                f"{key}/{name}": number
                for name, number in mean_metrics(value[:6], continuation=continuation).items()
            }
        )
        metrics[f"{key}/compression_ratio"] = float(value[6] / value[7])
        metrics[f"{key}/records"] = float(value[7])
    return metrics


@torch.no_grad()
def evaluate_teacher_forced(
    pipeline,
    qwen,
    adapter,
    cache,
    records,
    accelerator,
    *,
    overfit: bool = False,
    native=None,
) -> dict[str, float]:
    if adapter is not None:
        adapter = accelerator.unwrap_model(adapter)
        adapter.eval()
    qwen.model.eval()
    local = defaultdict(lambda: torch.zeros(8, dtype=torch.float64))
    cases = evaluation_cases(records, overfit=overfit)
    for index, (task, record, group) in enumerate(
        cases[accelerator.process_index :: accelerator.num_processes]
    ):
        with accelerator.autocast():
            if native is not None:
                hidden, targets, text_mask, visual_tokens = native.hidden(record, task)
            else:
                visual = cache.batch([record], "encoder", accelerator.device)
                adapted = adapter(visual).to(visual.dtype)
                hidden, targets, text_mask = qwen.student_hidden([record], adapted, task)
                visual_tokens = pipeline.config.adapter.sequence_length
            teacher = (
                cache.batch([record], "teacher", accelerator.device)
                if task == "continuation"
                else None
            )
            result = task_loss(
                hidden,
                targets,
                text_mask,
                head=qwen.model.lm_head,
                teacher=teacher,
                chunk_size=pipeline.config.training.loss_chunk_tokens,
            )
        values = torch.cat(
            [
                result.statistics.cpu(),
                torch.tensor(
                    [record["visual_token_count"] / visual_tokens, 1], dtype=torch.float64
                ),
            ]
        )
        for stratum in strata(record):
            key = group + (f"/{stratum}" if stratum else "")
            local[key] += values
        if (index + 1) % 100 == 0 and accelerator.is_main_process:
            print(f"evaluation: {index + 1} local records", flush=True)
    return aggregate_records({key: value.tolist() for key, value in local.items()}, accelerator)


def reconstruction_records(pipeline, records, *, final: bool):
    ordered = sorted(
        [row for row in records if row["split"] == "reconstruction"],
        key=lambda row: fingerprint([pipeline.config.seed, "generation", row["record_id"]]),
    )
    return ordered if final else ordered[: pipeline.config.evaluation.generation_subset]


@torch.no_grad()
def greedy_adapter(pipeline, qwen, adapter, cache, record) -> tuple[str, bool]:
    visual = cache.batch([record], "encoder", qwen.device)
    adapted = adapter(visual).to(visual.dtype)[0]
    inputs = torch.cat(
        [qwen.embed(qwen.reconstruction_before), adapted, qwen.embed(qwen.reconstruction_after)]
    )[None]
    qwen.model.eval()
    output = qwen.model.generate(
        inputs_embeds=inputs,
        attention_mask=torch.ones(inputs.shape[:2], device=qwen.device, dtype=torch.long),
        max_new_tokens=pipeline.config.evaluation.max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=qwen.assistant_end,
        pad_token_id=qwen.tokenizer.pad_token_id,
    )[0].tolist()
    stopped = bool(output and output[-1] == qwen.assistant_end)
    if stopped:
        output = output[:-1]
    return qwen.tokenizer.decode(
        output, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ), not stopped


@torch.no_grad()
def generate_reconstructions(
    pipeline,
    qwen,
    adapter,
    cache,
    records,
    accelerator,
    output_dir,
    step,
    *,
    final: bool,
    native=None,
) -> dict[str, float]:
    if adapter is not None:
        adapter = accelerator.unwrap_model(adapter)
        adapter.eval()
    chosen = reconstruction_records(pipeline, records, final=final)
    local = []
    for record in chosen[accelerator.process_index :: accelerator.num_processes]:
        with accelerator.autocast():
            prediction, truncated = (
                native.generate(record)
                if native is not None
                else greedy_adapter(pipeline, qwen, adapter, cache, record)
            )
        reference = record["visual"]
        characters = evaluate_edit_distance(reference, prediction, unit="character")
        words = evaluate_edit_distance(reference, prediction, unit="word")
        local.append(
            {
                "record_id": record["record_id"],
                "prediction": prediction,
                "reference": reference,
                "truncated": truncated,
                "character_distance": characters["distance"],
                "characters": characters["reference_units"],
                "word_distance": words["distance"],
                "words": words["reference_units"],
                "exact_match": prediction == reference,
            }
        )
    write_json(
        Path(output_dir) / f"predictions-{step:06d}-rank-{accelerator.process_index}.json", local
    )
    # Predictions stay local; distributed reduction exchanges only aggregate statistics.
    sums = torch.tensor(
        [
            len(local),
            sum(row["character_distance"] for row in local),
            sum(row["characters"] for row in local),
            sum(row["word_distance"] for row in local),
            sum(row["words"] for row in local),
            sum(row["exact_match"] for row in local),
            sum(row["truncated"] for row in local),
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    count, cd, chars, wd, words, exact, truncated = accelerator.reduce(
        sums, reduction="sum"
    ).tolist()
    if count != len(chosen):
        raise RuntimeError("generation evaluation record count mismatch")
    result = {
        "records": count,
        "character_edit_distance": cd / count,
        "cer": cd / max(chars, 1),
        "word_edit_distance": wd / count,
        "wer": wd / max(words, 1),
        "exact_match": exact / count,
        "truncation_rate": truncated / count,
    }
    if accelerator.is_main_process:
        write_json(Path(output_dir) / f"generation-{step:06d}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate optical adapters and frozen references")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", choices=("native", "teacher"))
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.reference):
        raise ValueError("provide exactly one of --checkpoint or --reference")
    pipeline = load_pipeline(args.config)
    load_credentials(pipeline, wandb=True)
    accelerator = Accelerator(mixed_precision="bf16")
    if accelerator.device.type != "cuda":
        raise ValueError("evaluation requires a selected CUDA device")
    records = load_manifest(pipeline)
    cache = TensorCache(pipeline, records)
    identity = fingerprint(
        {
            "data": pipeline.data_fingerprint,
            "models": pipeline.config.models.model_dump(),
            "reference": args.reference,
            "generation": args.generate,
        }
    )
    output = (
        args.checkpoint / "evaluation"
        if args.checkpoint
        else pipeline.output / "references" / args.reference
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.reference and (output / "complete.json").exists():
        previous = json.loads((output / "complete.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("reference evaluation fingerprint mismatch")
        if accelerator.is_main_process:
            print(json.dumps(previous["metrics"]), flush=True)
        return
    qwen = FrozenQwen(pipeline, accelerator.device)
    adapter, native = None, None
    if args.checkpoint:
        state = json.loads((args.checkpoint / "state.json").read_text())
        if state["resolved"]["data"] != pipeline.data_fingerprint:
            raise ValueError("checkpoint dataset fingerprint mismatch")
        adapter_config = json.loads((args.checkpoint / "adapter.json").read_text())
        if adapter_config["config"] != pipeline.config.adapter.model_dump():
            raise ValueError("checkpoint adapter configuration mismatch")
        adapter = build_adapter(adapter_config["kind"], pipeline.config.adapter).to(
            accelerator.device
        )
        adapter.load_state_dict(
            load_file(str(args.checkpoint / "adapter.safetensors")), strict=True
        )
    elif args.reference == "native":
        from optical_adaptor.training.native import NativeQwen

        native = NativeQwen(pipeline, qwen)
    if args.reference == "teacher":
        local = defaultdict(lambda: torch.zeros(8, dtype=torch.float64))
        cases = [
            (task, row, group)
            for task, row, group in evaluation_cases(records, overfit=False)
            if task == "continuation"
        ]
        for _, row, group in cases[accelerator.process_index :: accelerator.num_processes]:
            teacher = cache.batch([row], "teacher", accelerator.device).flatten(0, 1)
            targets = torch.tensor(row["continuation_ids"], device=accelerator.device)
            with torch.no_grad(), accelerator.autocast():
                result = task_loss(
                    teacher,
                    targets,
                    torch.ones_like(targets, dtype=torch.bool),
                    head=qwen.model.lm_head,
                    teacher=teacher,
                    chunk_size=pipeline.config.training.loss_chunk_tokens,
                )
            for stratum in strata(row):
                key = group + (f"/{stratum}" if stratum else "")
                local[key] += torch.cat(
                    [
                        result.statistics.cpu(),
                        torch.tensor([row["visual_token_count"] / 111, 1], dtype=torch.float64),
                    ]
                )
        metrics = aggregate_records(
            {key: value.tolist() for key, value in local.items()}, accelerator
        )
    else:
        metrics = evaluate_teacher_forced(
            pipeline, qwen, adapter, cache, records, accelerator, native=native
        )
    if args.generate and args.reference != "teacher":
        generation = generate_reconstructions(
            pipeline,
            qwen,
            adapter,
            cache,
            records,
            accelerator,
            output,
            0,
            final=True,
            native=native,
        )
        metrics.update({f"generation/{key}": value for key, value in generation.items()})
    if accelerator.is_main_process:
        from optical_adaptor.training.trainer import open_wandb

        run = open_wandb(
            pipeline,
            args.reference or adapter_config["kind"],
            "evaluation",
            {"data_fingerprint": pipeline.data_fingerprint, "reference": args.reference},
            None,
        )
        run.log(metrics)
        run.finish()
        write_json(output / "complete.json", {"identity": identity, "metrics": metrics})
        print(json.dumps(metrics), flush=True)
    accelerator.end_training()

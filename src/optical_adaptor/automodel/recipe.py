"""Optical data, evaluation and accounting hooks for the editable AutoModel KD recipe."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.vlm.kd import KnowledgeDistillationRecipeForVLM
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from optical_adaptor.automodel.config import OpticalConfig, fingerprint
from optical_adaptor.automodel.conversations import slice_key
from optical_adaptor.automodel.data import (
    ConversationDataset,
    MixtureSampler,
    inherited_values,
    removal_report,
    select_evaluation,
    validate_preparation,
)
from optical_adaptor.automodel.processing import OpticalProcessor
from optical_adaptor.edit_distance import levenshtein_distance
from optical_adaptor.renderer import load_render_config


class RunContract:
    def __init__(self, identity):
        self.identity = identity
        self.wandb_id = None
        self.consumed = Counter()

    def state_dict(self):
        return {
            "identity": self.identity,
            "wandb_id": self.wandb_id,
            "consumed": dict(self.consumed),
        }

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError(
                "Checkpoint model/data/loss/distributed contract does not match this run"
            )
        self.wandb_id = state["wandb_id"]
        self.consumed = Counter(state["consumed"])


class OpticalKDRecipe(KnowledgeDistillationRecipeForVLM):
    """Reuse framework setup, model builders, bridge, optimizer loop and checkpointing."""

    def __init__(self, cfg):
        load_dotenv()
        super().__init__(cfg)
        self.raw = cfg.to_yaml_dict()
        self.optical = OpticalConfig.model_validate(self.raw["optical"])
        for name, role in (("model", "student"), ("teacher_model", "teacher")):
            model = self.raw[name]
            expected = {
                "llm": self.optical.llm,
                "vision": self.optical.vision,
                "adapter": self.optical.adapter,
                "role": role,
                "image_microbatch_size": self.optical.processing.image_microbatch_size,
                "pretrained_model_name_or_path": self.optical.llm["model_id"],
            }
            for key, value in expected.items():
                if model[key] != value:
                    raise ValueError(f"{name}.{key} differs from the canonical optical config")
        if not 0 <= self.raw["kd_ratio"] <= 1 or self.raw["kd_loss_fn"]["chunk_size"] != 0:
            raise ValueError("KD ratio must be in [0,1] and position chunking must be disabled")
        self.events = {}

    def setup(self):
        super().setup()
        if self._should_setup_training_components():
            model = self.student_model()
            model.stage_timer = self._stage_timer
            if any(p.requires_grad for p in model.resources.language.parameters()):
                raise AssertionError("Backbone must be frozen")

    def student_model(self):
        model = self.model_parts[0]
        return (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )

    def _checkpoint_model(self, model):
        return [self.student_model().adapter]

    def _build_dataloaders(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.optical.llm["model_id"], revision=self.optical.llm["revision"]
        )
        self.processor = OpticalProcessor(
            tokenizer,
            self.optical.processing,
            load_render_config(self.optical.render_config),
            self.optical.vision,
        )
        self.compiler = self.processor.compiler
        validate_preparation(self.optical, self.cfg.seed)
        self.train_data, train_report = self._data("train")
        self.eval_data, eval_report = self._data("eval")
        self.sampler = MixtureSampler(
            self.train_data,
            self.optical.data,
            self.cfg.seed,
            self._get_dp_rank(),
            self._get_dp_group_size(),
            self.raw["step_scheduler"]["global_batch_size"],
        )
        self.untrack_state(
            "sampler"
        )  # StatefulDataLoader owns its cursor, including worker prefetch.
        self.dataloader = StatefulDataLoader(
            self.train_data,
            batch_size=self.raw["step_scheduler"]["local_batch_size"],
            sampler=self.sampler,
            collate_fn=self.processor,
            num_workers=self.optical.data.num_workers,
        )
        self.val_dataloader = None
        if len(self.eval_data):
            dp, rank = self._get_dp_group_size(), self._get_dp_rank()
            # All student replicas issue the same number of bridge requests. Padded
            # requests are discarded by evaluation, never counted as held-out samples.
            indices = [
                min(i + rank, len(self.eval_data) - 1) for i in range(0, len(self.eval_data), dp)
            ]
            self.val_dataloader = DataLoader(
                self.eval_data,
                batch_size=1,
                sampler=indices,
                collate_fn=self.processor,
                num_workers=self.optical.data.num_workers,
            )
        leaves = self.eval_data.all_leaves
        self.generation_limits = inherited_values(
            self.optical.evaluation.max_new_tokens, leaves, integer=True
        )
        if any(value < 1 for value in self.generation_limits.values()):
            raise ValueError("Generation output caps must be positive")
        if self.dist_env.is_main:
            output = Path(self.cfg.checkpoint.checkpoint_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "data-report.json").write_text(
                json.dumps(
                    {
                        "train": train_report,
                        "eval": eval_report,
                        "generation": eval_report["generation_quotas"],
                        "mixture": {
                            key: {"ratio": ratio, "eligible_rows": self.sampler.counts[key]}
                            for key, ratio in self.sampler.ratios.items()
                        },
                        "eval_ids": sorted(self.teacher_forced_ids),
                        "generation_ids": sorted(self.generation_ids),
                        "validation_scope": "sample-level",
                    },
                    indent=2,
                )
                + "\n"
            )

    def _data(self, split):
        filters = (
            self.optical.data.train_filter if split == "train" else self.optical.data.eval_filter
        )
        dataset = ConversationDataset(self.optical.prepare.output_dir, split, filters)
        before = list(dataset.rows)
        assigned = range(self._get_dp_rank(), len(dataset), self._get_dp_group_size())
        local = dataset.preflight(self.compiler, assigned)
        shards = [None] * self._get_dp_group_size()
        dist.all_gather_object(shards, local, group=self._get_dp_group())
        indices, dropped, lengths = [], Counter(), {}
        for selected, report in shards:
            indices.extend(selected)
            dropped.update(report["dropped"])
            lengths.update(report["lengths"])
        dataset.select(sorted(indices))
        report = {
            "filters": dataset.filter_report,
            "preflight": removal_report(before, dataset.rows, split + " preflight"),
            "dropped_reasons": dict(dropped),
            "lengths": lengths,
        }
        if not len(dataset) and split == "train":
            raise ValueError("No eligible training data")
        if split == "eval":
            all_requested = inherited_values(
                self.optical.data.eval_samples, dataset.all_leaves, integer=True
            )
            requested = {key: all_requested[key] for key in dataset.selected_leaves}
            indices, report["quotas"] = select_evaluation(dataset.rows, requested, self.cfg.seed)
            self.teacher_forced_ids = {dataset.rows[i]["sample_id"] for i in indices}
            generation_counts = inherited_values(
                self.optical.evaluation.generation_samples, dataset.all_leaves, integer=True
            )
            generation_requested = {
                key: generation_counts[key]
                if key.split("/")[1] in self.optical.evaluation.tasks
                else 0
                for key in dataset.selected_leaves
            }
            generation_indices, report["generation_quotas"] = select_evaluation(
                dataset.rows, generation_requested, self.cfg.seed
            )
            self.generation_ids = {dataset.rows[i]["sample_id"] for i in generation_indices}
            dataset.select(sorted(set(indices) | set(generation_indices)))
        report["selected_slices"] = dict(Counter(row["slice"] for row in dataset.rows))
        return dataset, report

    def _setup_run_state(self):
        identity = {
            key: self.raw[key]
            for key in (
                "model",
                "teacher_model",
                "distributed",
                "separate_meshes",
                "kd_ratio",
                "kd_loss_fn",
                "optimizer",
                "lr_scheduler",
                "clip_grad_norm",
                "optical",
            )
        }
        identity.update(
            teacher_distributed=self.raw.get("teacher_distributed"),
            train_ids=self.train_data.rows,
            eval_ids=self.eval_data.rows,
            world_size=self.dist_env.world_size,
            seed=self.cfg.seed,
            local_batch=self.raw["step_scheduler"]["local_batch_size"],
            global_batch=self.raw["step_scheduler"]["global_batch_size"],
            template=self.compiler.template,
        )
        self.contract = RunContract(fingerprint(identity))
        output = Path(self.cfg.checkpoint.checkpoint_dir)
        restore = self.cfg.get("checkpoint.restore_from", None)
        if restore is not None and not self.cfg.checkpoint.enabled:
            raise ValueError("Checkpointing must be enabled to restore a run")
        occupied = any(output.glob("epoch_*_step_*"))
        if restore == "LATEST" and not occupied:
            raise FileNotFoundError(f"No checkpoint in {output}")
        if restore is None and occupied:
            raise FileExistsError("Checkpoint directory is occupied; configure restore_from")

    def _setup_wandb(self):
        if self.cfg.wandb is not None and self.contract.wandb_id:
            self.cfg.wandb.extra.update(id=self.contract.wandb_id, resume="must")
        super()._setup_wandb()
        if wandb.run is not None:
            self.contract.wandb_id = wandb.run.id

    def save_checkpoint(self, epoch, step, *args, **kwargs):
        super().save_checkpoint(epoch, step, *args, **kwargs)
        if self.dist_env.is_main and self.cfg.checkpoint.enabled:
            from safetensors.torch import save_file

            path = Path(self.cfg.checkpoint.checkpoint_dir) / "exports" / f"step-{step + 1:06d}"
            path.mkdir(parents=True, exist_ok=True)
            state = {
                key: value.detach().cpu().contiguous()
                for key, value in self.student_model().adapter.state_dict().items()
            }
            save_file(state, path / "adapter.safetensors")
            (path / "optical-model.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "step": step + 1,
                        "model": {
                            "llm": self.optical.llm,
                            "vision": self.optical.vision,
                            "adapter": self.optical.adapter,
                        },
                        "run_contract": self.contract.identity,
                    },
                    indent=2,
                )
                + "\n"
            )

    @contextmanager
    def _stage_timer(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.setdefault(name, []).append((start, end))

    @torch.no_grad()
    def _record_kd_metrics(self, student, teacher, labels, ce_loss, kd_loss, denominator):
        valid = labels != -100
        count = valid.sum()
        teacher_ce = F.cross_entropy(
            teacher.flatten(0, 1).float(), labels.flatten(), reduction="sum"
        )
        student_ce = ce_loss.detach() * denominator
        if self.kd_ratio == 1:
            student_ce = F.cross_entropy(
                student.flatten(0, 1).float(), labels.flatten(), reduction="sum"
            )
        self.last_stats = torch.stack(
            (
                student_ce,
                kd_loss.detach() * denominator,
                teacher_ce,
                ((student.argmax(-1) == teacher.argmax(-1)) & valid).sum(),
                ((student.argmax(-1) == labels) & valid).sum(),
                count,
            )
        )

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        self.events.clear()
        started = time.perf_counter()
        self.timestamp = started
        result = super()._run_train_optim_step(batches, max_grad_norm)
        torch.cuda.synchronize()  # One collection boundary, never synchronize each substep.
        elapsed = self._dp_allreduce(
            torch.tensor(time.perf_counter() - started + self.batch_wait_seconds),
            op=dist.ReduceOp.MAX,
        ).item()
        local_counts = Counter(
            slice_key(record) for batch in batches for record in batch["records"]
        )
        counts = self._dp_allreduce(
            torch.tensor([local_counts[key] for key in self.sampler.ratios])
        ).tolist()
        metrics = {
            "timing/step": elapsed,
            "timing/data_wait": self._dp_allreduce(
                torch.tensor(self.batch_wait_seconds), op=dist.ReduceOp.MAX
            ).item(),
        }
        for key, count in zip(self.sampler.ratios, counts, strict=True):
            self.contract.consumed[key] += count
            metrics.update(
                {
                    f"data/ratio/{key}": self.sampler.ratios[key],
                    f"data/observed_ratio/{key}": count / sum(counts),
                    f"data/training_samples/{key}": self.contract.consumed[key],
                    f"data/training_epochs/{key}": self.contract.consumed[key]
                    / self.sampler.counts[key],
                }
            )
        totals = [sum(len(batch["records"]) for batch in batches)]
        pairs = [pair for batch in batches for pair in batch["compiled"]]
        totals += [
            sum(len(getattr(pair, field)) for pair in pairs)
            for field in ("teacher_ids", "student_ids", "targets", "image_positions")
        ]
        totals = self._dp_allreduce(torch.tensor(totals)).tolist()
        names = ("size", "teacher_input_tokens", "student_input_tokens", "loss_tokens", "images")
        metrics.update({f"batch/{name}": value for name, value in zip(names, totals, strict=True)})
        for name, value in zip(names[1:4], totals[1:4], strict=True):
            metrics[f"throughput/{name}_per_second"] = value / elapsed
        for name, events in self.events.items():
            seconds = sum(start.elapsed_time(end) for start, end in events) / 1000
            metrics[f"timing/{name}"] = self._dp_allreduce(
                torch.tensor(seconds), op=dist.ReduceOp.MAX
            ).item()
        metrics["tps"] = totals[3] / elapsed
        result.metrics.update(metrics)
        return result

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        for model in self.model_parts:
            model.eval()
        keys = sorted({row["slice"] for row in self.eval_data.rows})
        offsets = {key: i for i, key in enumerate(keys)}
        totals = torch.zeros((len(keys), 6), dtype=torch.float64, device=self.dist_env.device)
        generation = torch.zeros((len(keys), 6), dtype=torch.float64, device=self.dist_env.device)
        generation_ids = self.generation_ids
        generate = self.step_scheduler.is_last_step or (
            (self.step_scheduler.step + 1) % self.optical.evaluation.generation_every == 0
        )
        generation_rows = []
        for batch_index, batch in enumerate(val_dataloader):
            record = batch["records"][0]
            index = offsets[slice_key(record)]
            self._forward_backward_step(
                0,
                batch,
                loss_buffer=[],
                num_label_tokens=len(batch["compiled"][0].targets),
                num_batches=1,
                is_train=False,
            )
            valid = batch_index * self._get_dp_group_size() + self._get_dp_rank() < len(
                self.eval_data
            )
            if not valid:
                continue
            if record["sample_id"] in self.teacher_forced_ids:
                totals[index] += self.last_stats.double()
            if generate and record["sample_id"] in generation_ids:
                if record["thinking"]:
                    raise ValueError(
                        "Generative edit-distance evaluation requires a no-thinking task"
                    )
                pair = batch["compiled"][0]
                model = self.student_model()
                inputs = {
                    key: value.to(self.dist_env.device)
                    for key, value in batch["student"].items()
                    if key in {"input_ids", "attention_mask", "pixel_values", "image_positions"}
                }
                for key in ("input_ids", "attention_mask"):
                    inputs[key] = inputs[key][:, : pair.generation_prefix_length]
                ids = model.generate(
                    **inputs, max_new_tokens=self.generation_limits[slice_key(record)]
                )
                reached_limit = ids[0, -1].item() != model.tokenizer.convert_tokens_to_ids(
                    "<|im_end|>"
                )
                prediction = model.tokenizer.decode(ids[0], skip_special_tokens=True)
                reference = next(
                    m["content"] for m in reversed(record["messages"]) if m["role"] == "assistant"
                )
                char_edits = levenshtein_distance(reference, prediction)
                line_edits = levenshtein_distance(reference.splitlines(), prediction.splitlines())
                generation[index] += torch.tensor(
                    [
                        char_edits,
                        len(reference),
                        line_edits,
                        len(reference.splitlines()),
                        1,
                        int(reached_limit),
                    ],
                    device=self.dist_env.device,
                )
                generation_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "slice": slice_key(record),
                        "reference": reference,
                        "prediction": prediction,
                        "reached_generation_limit": reached_limit,
                    }
                )
        self._ce_loss_buffer.clear()
        self._kd_loss_buffer.clear()
        totals = self._dp_allreduce(totals)
        generation = self._dp_allreduce(generation)
        metrics = {}
        for key, sums in [
            *aggregate_metrics(keys, totals),
            *[("eval-aux/" + name, totals[i]) for i, name in enumerate(keys)],
        ]:
            count = sums[5].item()
            if count == 0:
                continue
            ce, kl, teacher_ce, agreement, correct = (sums[:5] / count).tolist()
            metrics.update(
                {
                    f"{key}/ce": ce,
                    f"{key}/kl": kl,
                    f"{key}/teacher_ce": teacher_ce,
                    f"{key}/agreement": agreement,
                    f"{key}/token_accuracy": correct,
                    f"{key}/loss": (1 - self.kd_ratio) * ce + self.kd_ratio * kl,
                }
            )
        for key, sums in [
            *aggregate_metrics(keys, generation),
            *[("eval-aux/" + name, generation[i]) for i, name in enumerate(keys)],
        ]:
            if sums[4] > 0:
                metrics.update(
                    {
                        f"{key}/cer": (sums[0] / sums[1].clamp_min(1)).item(),
                        f"{key}/ler": (sums[2] / sums[3].clamp_min(1)).item(),
                        f"{key}/generated_samples": sums[4].item(),
                        f"{key}/character_edits": sums[0].item(),
                        f"{key}/line_edits": sums[2].item(),
                        f"{key}/generation_limit_fraction": (sums[5] / sums[4]).item(),
                    }
                )
        if generation_rows:
            path = Path(self.cfg.checkpoint.checkpoint_dir) / (
                f"generations-step-{self.step_scheduler.step}-rank-{self._get_dp_rank()}.jsonl"
            )
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generation_rows),
                encoding="utf-8",
            )
        # val_loss is the AutoModel loop's checkpoint metric.
        if "eval-core/loss" in metrics:
            metrics["val_loss"] = metrics["eval-core/loss"]
        return MetricsSample(
            step=self.step_scheduler.step, epoch=self.step_scheduler.epoch, metrics=metrics
        )

    def log_val_metrics(self, log_data):
        if self.dist_env.is_main:
            self.metric_logger_valid.log(log_data)
            if wandb.run is not None:
                wandb.log(log_data.metrics, step=log_data.step)
            logging.info(
                "Evaluation aggregate metrics: %s",
                {
                    key: value
                    for key, value in log_data.metrics.items()
                    if key.startswith("eval-core/") and key.count("/") == 1
                },
            )


def aggregate_metrics(keys, totals):
    result = [("eval-core", totals.sum(0))]
    for axis, level in (("source", 0), ("task", 1)):
        for name in sorted({key.split("/")[level] for key in keys}):
            indices = [i for i, key in enumerate(keys) if key.split("/")[level] == name]
            result.append((f"eval-core/{axis}/{name}", totals[indices].sum(0)))
    return result

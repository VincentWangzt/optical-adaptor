"""AutoModel VLM KD recipe with paired text/optical inputs and adapter-only updates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.config import DDPConfig
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.training.rng import ScopedRNG, StatefulRNG
from nemo_automodel.recipes.vlm.kd import KnowledgeDistillationRecipeForVLM
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from optical_adaptor.automodel.config import OpticalConfig, fingerprint, read_config
from optical_adaptor.automodel.conversations import slice_key
from optical_adaptor.automodel.data import ConversationDataset, MixtureSampler, validate_preparation
from optical_adaptor.automodel.model import FrozenBackbone, aligned_losses
from optical_adaptor.automodel.processing import (
    ConversationCompiler,
    collate_records,
    render_batch,
)
from optical_adaptor.edit_distance import levenshtein_distance
from optical_adaptor.renderer import load_render_config


class RunContract:
    """Checkpointed identity; incompatible model/data/loss resumes fail explicitly."""

    def __init__(self, identity: str):
        self.identity = identity
        self.wandb_id = None

    def state_dict(self):
        return {"identity": self.identity, "wandb_id": self.wandb_id}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError(
                "Checkpoint model/data/loss/distributed contract does not match this run"
            )
        self.wandb_id = state["wandb_id"]


class OpticalKDRecipe(KnowledgeDistillationRecipeForVLM):
    """Inherit AutoModel's accumulation, optimizer, scheduler, loop and checkpoint engine.

    Setup is specialized because the trainable model is an MLP, while the frozen
    original language model is shared by teacher and student on every DP rank.
    The forward step uses independent context positions and identical target IDs.
    """

    def __init__(self, raw: dict):
        super().__init__(ConfigNode(raw))
        self.optical = OpticalConfig.model_validate(raw["optical"])
        self.raw = raw

    def setup(self):
        if self.cfg.get("separate_meshes", False):
            raise ValueError("Optical KD currently requires a shared teacher/student setup")
        if not 0 <= self.cfg.kd_ratio <= 1:
            raise ValueError("kd_ratio must lie in [0, 1]")
        self.dist_env = initialize_distributed(**self.raw["dist_env"])
        setup_logging()
        self.rng = StatefulRNG(seed=self.cfg.seed, ranked=True)
        (
            self.distributed_setup,
            self.mesh_context,
            self.distributed_config,
            self.device_mesh,
            self.moe_mesh,
            self.pp_enabled,
            self.pipeline_config,
            self.moe_parallel_config,
            self.activation_checkpointing,
        ) = self._distributed_setup_attributes(self._create_distributed_setup())
        if not isinstance(self.distributed_config, DDPConfig):
            raise ValueError("This adapter-only recipe currently supports AutoModel DDP")
        if self.activation_checkpointing:
            raise ValueError("Configure checkpointing under optical.llm, not the small adapter")
        self.peft_config, self.pp = None, None
        self.render_config = load_render_config(self.optical.render_config)
        self.processor = AutoTokenizer.from_pretrained(
            self.optical.llm["model_id"], revision=self.optical.llm["revision"]
        )
        self.compiler = ConversationCompiler(
            self.processor, self.optical.processing, self.optical.vision["tokens_per_image"]
        )
        output = Path(self.cfg.checkpoint.checkpoint_dir)
        output.mkdir(parents=True, exist_ok=True)
        validate_preparation(self.optical, self.cfg.seed)
        self.train_data, train_report = self._data("train")
        self.eval_data, eval_report = self._data("eval")
        self.contract = RunContract(
            fingerprint(
                {
                    "optical": self.optical.model_dump(exclude={"evaluation"}),
                    "training_rows": self.train_data.rows,
                    "eval_rows": self.eval_data.rows,
                    "render": Path(self.optical.render_config).read_text(),
                    "template": self.compiler.template,
                    "world_size": self.dist_env.world_size,
                    "seed": self.cfg.seed,
                    "kd_ratio": self.cfg.kd_ratio,
                    "kd_loss": self.raw["kd_loss_fn"],
                    "optimizer": self.raw["optimizer"],
                    "lr_scheduler": self.raw["lr_scheduler"],
                    "clip_grad_norm": self.raw["clip_grad_norm"],
                    "global_batch_size": self.raw["step_scheduler"]["global_batch_size"],
                    "local_batch_size": self.raw["step_scheduler"]["local_batch_size"],
                }
            )
        )
        if self.dist_env.is_main:
            (output / "data-report.json").write_text(
                json.dumps(
                    {
                        "train": train_report,
                        "eval": eval_report,
                        "contract": self.contract.identity,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        sampler = MixtureSampler(
            self.train_data,
            self.optical.prepare.sources,
            self.optical.data,
            self.cfg.seed,
            self._get_dp_rank(),
            self._get_dp_group_size(),
        )
        collate = partial(collate_records, compiler=self.compiler)
        self.dataloader = StatefulDataLoader(
            self.train_data,
            batch_size=self.raw["step_scheduler"]["local_batch_size"],
            sampler=sampler,
            collate_fn=collate,
            num_workers=self.optical.data.num_workers,
        )
        # Disjoint evaluation without DistributedSampler padding/repeated examples.
        self.val_dataloader = DataLoader(
            self.eval_data,
            batch_size=1,
            sampler=list(
                range(self._get_dp_rank(), len(self.eval_data), self._get_dp_group_size())
            ),
            collate_fn=collate,
            num_workers=self.optical.data.num_workers,
        )
        self.backbone = FrozenBackbone(
            self.optical.llm, self.dist_env.device, self.optical.adapter["output_dim"]
        )
        self.teacher_model = self.backbone.model
        self.vision = (
            ConfigNode(self.optical.vision)
            .instantiate()
            .to(device=self.dist_env.device, dtype=torch.bfloat16)
        )
        self.untrack_state("vision")
        adapter = self.cfg.model.instantiate().to(self.dist_env.device, dtype=torch.float32)
        adapter = DDPManager(self.distributed_config).parallelize(adapter)
        if self.dist_env.world_size == 1:
            adapter.float()  # DDPManager's single-device branch casts weights to BF16.
        self.model_parts = [adapter]
        self.optimizer = self.cfg.optimizer.build(adapter, device_mesh=self.device_mesh)
        self.checkpointer = self.cfg.checkpoint.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=0,
            pp_rank=0,
        )
        self.step_scheduler = self.cfg.step_scheduler.build(
            self.dataloader,
            self._get_dp_group_size(),
            self.raw["step_scheduler"]["local_batch_size"],
        )
        self._setup_garbage_collection(self.step_scheduler)
        self.lr_scheduler = self.cfg.lr_scheduler.build(self.optimizer, self.step_scheduler)
        self.max_grad_norm = self.cfg.clip_grad_norm.max_norm
        self.best_metric_key = "default"
        self.loss_fn = self.cfg.loss_fn.build()
        self._setup_kd_state()
        self.metric_logger_train = build_metric_logger(output / "training.jsonl")
        self.metric_logger_valid = build_metric_logger(output / "validation.jsonl")
        restore = self.raw["checkpoint"]["restore_from"]
        if restore is not None:
            if not self.cfg.checkpoint.enabled:
                raise ValueError("Restoring a run requires checkpoint.enabled=true")
            if restore == "LATEST" and not any(output.glob("epoch_*_step_*")):
                raise FileNotFoundError(f"No checkpoint to restore in {output}")
            self.load_checkpoint(restore)
        elif any(output.glob("epoch_*_step_*")):
            raise FileExistsError("Checkpoint directory is occupied; set restore_from explicitly")
        if self.dist_env.is_main and self.cfg.wandb is not None:
            mode = self.cfg.wandb.extra.get("mode", "online")
            if mode == "online" and not os.environ.get("WANDB_API_KEY"):
                raise ValueError("Online W&B logging requires WANDB_API_KEY (load it from .env)")
            if self.contract.wandb_id:
                self.cfg.wandb.extra.update(id=self.contract.wandb_id, resume="must")
            with ScopedRNG(seed=self.cfg.seed, ranked=True):
                run = self.cfg.wandb.build(run_config=self.raw, model_name="optical-adaptor")
            self.contract.wandb_id = run.id
            logging.info("W&B: %s", run.url)
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        if any(p.requires_grad for p in self.teacher_model.parameters()):
            raise AssertionError("Language model must be entirely frozen")
        if any(p.requires_grad for p in self.vision.parameters()):
            raise AssertionError("Vision encoder must be entirely frozen")
        logging.info("Trainable adapter parameters: %d; frozen encoder and backbone", trainable)

    def _data(self, split):
        filters = (
            self.optical.data.train_filter if split == "train" else self.optical.data.eval_filter
        )
        dataset = ConversationDataset(self.optical.prepare.output_dir, split, filters, None)
        payload = [dataset.preflight(self.compiler) if self.dist_env.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        indices, report = payload[0]
        dataset.select(indices)
        if split == "eval" and self.optical.data.eval_samples_per_slice is not None:
            counts, chosen = Counter(), []
            for index, row in enumerate(dataset.rows):
                if counts[row["slice"]] < self.optical.data.eval_samples_per_slice:
                    chosen.append(index)
                    counts[row["slice"]] += 1
            dataset.select(chosen)
        report["selected_slices"] = dict(Counter(row["slice"] for row in dataset.rows))
        logging.info("%s: %d selected examples; %s dropped", split, len(dataset), report["dropped"])
        return dataset, report

    def _adapt(self, pairs, is_train):
        images = render_batch(pairs, self.render_config)
        features = []
        size = self.optical.processing.image_microbatch_size
        for start in range(0, len(images), size):
            pixels = self.vision.pixels(images[start : start + size])
            features.append(self.vision(pixels))
        features = torch.cat(features)
        model = self.model_parts[0]
        if not is_train and isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module  # Uneven eval shards must not issue DDP collectives.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(features)

    def _forward_backward_step(
        self, idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True
    ):
        model = self.model_parts[0]
        sync = (
            get_sync_ctx(model, idx == num_batches - 1, defer_fsdp_grad_sync=False)
            if is_train
            else nullcontext()
        )
        with sync:
            pairs = batch["compiled"]
            teacher = self.backbone.teacher_hidden(pairs)
            adapted = self._adapt(pairs, is_train)
            student = self.backbone.student_hidden(pairs, adapted)
            targets = torch.tensor(
                [token for pair in pairs for token in pair.targets], device=self.dist_env.device
            )
            stats = aligned_losses(
                student,
                teacher,
                targets,
                self.teacher_model.get_output_embeddings(),
                self.kd_loss_fn,
                self.optical.processing.loss_chunk_tokens,
            )
            ce, kl = stats[:2] / num_label_tokens
            loss = (1 - self.kd_ratio) * ce + self.kd_ratio * kl
            loss_buffer.append(loss.detach())
            self._ce_loss_buffer.append(ce.detach())
            self._kd_loss_buffer.append(kl.detach())
            self.last_stats = stats.detach()
            if is_train:
                (loss * self._get_dp_group_size()).backward()
                if any(p.grad is not None for p in self.teacher_model.parameters()):
                    raise AssertionError("Frozen backbone received parameter gradients")
                if any(p.grad is not None for p in self.vision.parameters()):
                    raise AssertionError("Frozen encoder received parameter gradients")

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        for model in self.model_parts:
            model.eval()
        keys = sorted({row["slice"] for row in self.eval_data.rows})
        offsets = {key: i for i, key in enumerate(keys)}
        totals = torch.zeros((len(keys), 6), dtype=torch.float64, device=self.dist_env.device)
        generation = torch.zeros((len(keys), 6), dtype=torch.float64, device=self.dist_env.device)
        gen_counts = Counter()
        generation_ids = set()
        for row in self.eval_data.rows:
            if row["task"] in self.optical.evaluation.tasks and (
                gen_counts[row["slice"]] < self.optical.evaluation.generation_samples_per_slice
            ):
                gen_counts[row["slice"]] += 1
                generation_ids.add(row["sample_id"])
        generate = self.step_scheduler.is_last_step or (
            (self.step_scheduler.step + 1) % self.optical.evaluation.generation_every == 0
        )
        generation_rows = []
        for batch in val_dataloader:
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
            totals[index] += self.last_stats.double()
            if generate and record["sample_id"] in generation_ids:
                if record["thinking"]:
                    raise ValueError(
                        "Generative edit-distance evaluation requires a no-thinking task"
                    )
                pair = batch["compiled"][0]
                adapted = self._adapt([pair], False)
                prediction, reached_limit = self.backbone.generate(
                    pair, adapted, self.optical.evaluation.max_new_tokens
                )
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
        dist.all_reduce(totals)
        dist.all_reduce(generation)
        metrics = {}
        for key, sums in [
            ("eval-core", totals.sum(0)),
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
                    f"{key}/tokens": count,
                    f"{key}/loss": (1 - self.kd_ratio) * ce + self.kd_ratio * kl,
                }
            )
        for key, sums in [
            ("eval-core", generation.sum(0)),
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
                "Evaluation CE=%.5f KL=%.5f",
                log_data.metrics["eval-core/ce"],
                log_data.metrics["eval-core/kl"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/automodel.yaml")
    args = parser.parse_args()
    load_dotenv()
    raw, _ = read_config(args.config)
    recipe = OpticalKDRecipe(raw)
    try:
        recipe.setup()
        recipe.run_train_validation_loop()
    finally:
        if wandb.run is not None:
            wandb.finish(exit_code=int(sys.exc_info()[0] is not None))
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

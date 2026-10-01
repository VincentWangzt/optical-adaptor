"""Continue optical KD from a full checkpoint with explicit generation tasks.

Launch with ``uv run --no-sync torchrun --standalone --nproc-per-node=N``.
The optimizer and scheduler are restored. The deterministic global sampler
stream resumes at the consumed-sample offset, redistributed across the new DP
ranks. An unchanged topology restores rank-local RNG and dataloader state;
a changed topology uses newly seeded rank-local RNG streams.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
import yaml
from nemo_automodel.components.config.loader import ConfigNode
from safetensors.torch import load_file

from optical_adaptor.automodel.config import OpticalConfig
from optical_adaptor.automodel.recipe import OpticalKDRecipe


class ContinuedOpticalKDRecipe(OpticalKDRecipe):
    def __init__(self, cfg, checkpoint: Path, adapter: Path, source_world_size: int):
        self.source_checkpoint = checkpoint
        self.source_adapter = adapter
        self.source_world_size = source_world_size
        super().__init__(cfg)

    def load_checkpoint(self, restore_from=None):
        if restore_from is not None:
            raise ValueError("The continuation loads its source checkpoint explicitly")
        state = torch.load(self.source_checkpoint / "step_scheduler.pt", weights_only=True)
        step, epoch = state["step"], state["epoch"]
        if epoch != 0 or step != self.raw["step_scheduler"]["start_step"]:
            raise ValueError("Source scheduler does not match the configured continuation step")
        contract = torch.load(self.source_checkpoint / "contract.pt", weights_only=True)
        global_batch = self.raw["step_scheduler"]["global_batch_size"]
        consumed = step * global_batch
        if sum(contract["consumed"].values()) != consumed:
            raise ValueError("Source sample accounting does not match the scheduler step")
        old_cursor = consumed // self.source_world_size
        for rank in range(self.source_world_size):
            path = self.source_checkpoint / "dataloader" / f"dataloader_dp_rank_{rank}.pt"
            dataloader = torch.load(path, weights_only=True)
            snapshot = dataloader["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]
            if snapshot["sampler_state"] != {"epoch": epoch, "cursor": old_cursor}:
                raise ValueError(f"Source sampler cursor differs on rank {rank}")

        world_size = self._get_dp_group_size()
        if consumed % world_size:
            raise ValueError("Consumed sample offset must divide the new DP size")
        self.student_model().adapter.load_state_dict(
            load_file(self.source_adapter, device="cpu"), strict=True
        )
        self.checkpointer.load_optimizer(
            self.optimizer,
            self._checkpoint_model(self.model_parts),
            str(self.source_checkpoint),
            self.lr_scheduler,
            optimizer_part_ids=self._get_optimizer_checkpoint_part_ids(),
        )
        optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
        optimizer_steps = {
            int(value["step"])
            for optimizer in optimizers
            for value in optimizer.state.values()
            if "step" in value
        }
        if optimizer_steps != {step}:
            raise ValueError(f"Restored AdamW step counts differ: {optimizer_steps}")
        self.step_scheduler.load_state_dict(state)
        self.sampler.load_state_dict({"epoch": epoch, "cursor": consumed // world_size})
        self.contract.consumed = Counter(contract["consumed"])
        if world_size == self.source_world_size:
            self.checkpointer.load_on_dp_ranks(
                self.dataloader, "dataloader", str(self.source_checkpoint)
            )
            self.checkpointer.load_on_global_ranks(self.rng, "rng", str(self.source_checkpoint))
            rng_policy = "restored checkpoint rank-local RNG and dataloader state"
        else:
            rng_policy = "new rank-seeded streams; source RNG is not portable across DP sizes"
        if self.dist_env.is_main:
            output = Path(self.cfg.checkpoint.checkpoint_dir).parent
            (output / "continuation.json").write_text(
                json.dumps(
                    {
                        "source_checkpoint": str(self.source_checkpoint.resolve()),
                        "source_adapter": str(self.source_adapter.resolve()),
                        "source_step": step,
                        "source_world_size": self.source_world_size,
                        "new_world_size": world_size,
                        "global_samples_consumed": consumed,
                        "new_rank_sampler_cursor": self.sampler.cursor,
                        "optimizer_steps": sorted(optimizer_steps),
                        "source_contract_identity": contract["identity"],
                        "new_contract_identity": self.contract.identity,
                        "rng_policy": rng_policy,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"Continued step {step}, optimizer step {step}, "
                f"global sample offset {consumed}, DP {self.source_world_size}->{world_size}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--generation-samples-per-slice", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, required=True)
    parser.add_argument(
        "--generation-tasks", nargs="+", choices=["reconstruction", "continuation"], required=True
    )
    args = parser.parse_args()
    source = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    saved = yaml.safe_load((args.checkpoint / "config.yaml").read_text(encoding="utf-8"))
    if source != saved:
        raise ValueError("Source config differs from the checkpoint's saved config")
    topology = source["distributed"]
    if topology["strategy"] != "ddp" or source["separate_meshes"]:
        raise ValueError("Only single-mesh DDP checkpoints can change DP size")
    if any(topology[key] != 1 for key in ("tp_size", "cp_size", "pp_size", "ep_size")):
        raise ValueError("Source TP, CP, PP and EP sizes must all be one")
    source_ranks = sorted((args.checkpoint / "dataloader").glob("dataloader_dp_rank_*.pt"))
    if not source_ranks:
        raise FileNotFoundError("Source checkpoint has no per-rank dataloader states")
    source_world_size = len(source_ranks)
    step = source["step_scheduler"]["max_steps"]
    if args.max_steps <= step:
        raise ValueError("New max_steps must exceed the source step")
    if args.generation_samples_per_slice < 1 or args.generation_batch_size < 1:
        raise ValueError("Generation sample and batch sizes must be positive")
    exported = json.loads((args.adapter.parent / "optical-model.json").read_text())
    if exported["step"] != step or exported["model"] != {
        key: source["optical"][key] for key in ("llm", "vision", "adapter")
    }:
        raise ValueError("Adapter export does not match the source step and model")

    rank = int(os.environ["RANK"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0 and any(output.iterdir()):
        raise FileExistsError(f"Continuation output directory is occupied: {output}")
    source["step_scheduler"]["max_steps"] = args.max_steps
    source["step_scheduler"]["start_step"] = step
    source["checkpoint"].update(checkpoint_dir=str(output / "checkpoints"), restore_from=None)
    source["wandb"]["name"] = output.name
    source["optical"]["evaluation"].update(
        generation_samples={"": args.generation_samples_per_slice},
        generation_batch_size=args.generation_batch_size,
        tasks=args.generation_tasks,
    )
    OpticalConfig.model_validate(source["optical"])
    if rank == 0:
        (output / "config.yaml").write_text(
            yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
        )
        (output / "git-commit.txt").write_text(
            os.environ.get("GIT_COMMIT", "unknown") + "\n", encoding="utf-8"
        )

    trainer = ContinuedOpticalKDRecipe(
        ConfigNode(source), args.checkpoint, args.adapter, source_world_size
    )
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()

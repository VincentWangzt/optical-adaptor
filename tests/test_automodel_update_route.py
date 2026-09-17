"""Opt-in full-model diagnostic: run with torchrun on one explicitly selected GPU.

Uses fixed processed batches and restores adapter/optimizer/RNG state between
trials. It records forward hashes, pre-clipping gradients and Adam update checks.
This is a diagnostic entry point, not part of default pytest collection.
"""

import argparse
import copy
import hashlib
import json
import traceback
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import yaml
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.vlm import kd
from torch.nn.attention import SDPBackend, sdpa_kernel

from optical_adaptor.automodel.recipe import OpticalKDRecipe


def digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def inspect_updates(config_path, output_dir):
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["separate_meshes"] = False
    raw["distributed"]["dp_size"] = 1
    raw.pop("teacher_distributed", None)
    raw["wandb"] = None
    raw["checkpoint"].update(enabled=False, restore_from=None, checkpoint_dir=str(output_dir))
    raw["optical"]["data"]["num_workers"] = 0
    raw["optical"]["data"]["eval_samples"] = {"": 0}
    raw["optical"]["evaluation"]["generation_samples"] = {"": 0}
    recipe = OpticalKDRecipe(ConfigNode(raw))
    recipe.setup()
    model, optimizer = recipe.student_model(), recipe.optimizer[0]
    model.train()
    parameters = dict(model.adapter.named_parameters())
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len(parameters)
    assert {id(p) for p in optimized} == {id(p) for p in parameters.values()}
    assert all(p.dtype == torch.float32 for p in optimized)
    assert all(not p.requires_grad for p in model.resources.parameters())
    frozen_versions = {name: p._version for name, p in model.resources.named_parameters()}

    iterator = iter(recipe.dataloader)
    count = raw["step_scheduler"]["global_batch_size"] // raw["step_scheduler"]["local_batch_size"]
    batches = [next(iterator) for _ in range(count)]
    initial = copy.deepcopy(model.adapter.state_dict())
    optimizer_initial = copy.deepcopy(optimizer.state_dict())
    scheduler_initial = [copy.deepcopy(s.state_dict()) for s in recipe.lr_scheduler]
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    original_clip = kd.scale_grads_and_clip_grad_norm
    results, comparisons = {}, {}
    last = {}

    for name, deterministic, checkpointing in (
        ("default_a", False, True),
        ("default_b", False, True),
        ("strict_math_a", True, True),
        ("strict_math_b", True, True),
        ("strict_math_no_checkpoint", True, False),
    ):
        model.adapter.load_state_dict(initial)
        optimizer.load_state_dict(copy.deepcopy(optimizer_initial))
        optimizer.zero_grad(set_to_none=True)
        for scheduler, state in zip(recipe.lr_scheduler, scheduler_initial, strict=True):
            scheduler.load_state_dict(copy.deepcopy(state))
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)
        torch.use_deterministic_algorithms(deterministic)
        if checkpointing:
            model.resources.language.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.resources.language.gradient_checkpointing_disable()
        recipe.batch_wait_seconds = 0
        captured = {"teacher": [], "student": []}

        def capture_clip(*args, capture=captured, **kwargs):
            capture["raw_gradient"] = torch.cat(
                [p.grad.detach().flatten().cpu() for p in parameters.values()]
            )
            return original_clip(*args, **kwargs)

        def capture_update(opt, args, kwargs, capture=captured):
            reference = copy.deepcopy(model.adapter)
            reference_optimizer = torch.optim.AdamW(reference.parameters())
            reference_optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
            for expected, actual in zip(reference.parameters(), parameters.values(), strict=True):
                expected.grad = actual.grad.detach().clone()
            reference_optimizer.step()
            capture["expected_update"] = copy.deepcopy(reference.state_dict())
            capture["expected_optimizer"] = copy.deepcopy(reference_optimizer.state_dict())

        handles = [
            model.register_forward_hook(
                lambda module, args, value, capture=captured: capture["student"].append(
                    digest(value.logits)
                )
            ),
            recipe.teacher_model.register_forward_hook(
                lambda module, args, value, capture=captured: capture["teacher"].append(
                    digest(value.logits)
                )
            ),
            optimizer.register_step_pre_hook(capture_update),
        ]
        attention = sdpa_kernel(SDPBackend.MATH) if deterministic else nullcontext()
        try:
            with attention, patch.object(kd, "scale_grads_and_clip_grad_norm", capture_clip):
                metrics = recipe._run_train_optim_step(batches, recipe.max_grad_norm).metrics
            update_error = max(
                (value - captured["expected_update"][key]).abs().max().item()
                for key, value in model.adapter.state_dict().items()
            )
            actual_state = optimizer.state_dict()["state"]
            expected_state = captured["expected_optimizer"]["state"]
            moment_error = max(
                (value - expected_state[index][key]).abs().max().item()
                for index, state in actual_state.items()
                for key, value in state.items()
            )
            assert update_error == 0 and moment_error == 0
            assert all(p.grad is None for p in optimized)
            assert all(p.grad is None for p in model.resources.parameters())
            assert frozen_versions == {
                key: p._version for key, p in model.resources.named_parameters()
            }
            results[name] = {
                "loss": metrics["loss"],
                "grad_norm": float(metrics["grad_norm"]),
                "teacher_hashes": captured["teacher"],
                "student_hashes": captured["student"],
                "adam_parameter_error": update_error,
                "adam_state_error": moment_error,
            }
            gradients = captured["raw_gradient"]
            updated = torch.cat([p.detach().flatten().cpu() for p in parameters.values()])
            for previous, (old_gradients, old_updated) in last.items():
                comparisons[f"{previous}:{name}"] = {
                    "raw_gradient_max_difference": (gradients - old_gradients).abs().max().item(),
                    "raw_gradient_relative_l2": (
                        (gradients - old_gradients).norm() / old_gradients.norm()
                    ).item(),
                    "gradient_sign_changes": int(((gradients * old_gradients) < 0).sum()),
                    "updated_parameter_max_difference": (updated - old_updated).abs().max().item(),
                }
            last[name] = (gradients, updated)
        except RuntimeError as error:
            # Keep backend determinism errors visible in this diagnostic report.
            results[name] = {"error": str(error), "traceback": traceback.format_exc()}
            optimizer.zero_grad(set_to_none=True)
            recipe._ce_loss_buffer.clear()
            recipe._kd_loss_buffer.clear()
        finally:
            for handle in handles:
                handle.remove()
        report = {"trials": results, "comparisons": comparisons}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "update-route.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: results[name]}), flush=True)

    recipe.metric_logger_train.close()
    recipe.metric_logger_valid.close()
    recipe._finalize_and_close_checkpointer()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect_updates(args.config, args.output)

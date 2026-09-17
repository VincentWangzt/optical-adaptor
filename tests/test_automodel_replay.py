"""Compare full-run checkpoint artifacts after a fresh-process resumed update."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def compare_tree(left, right, path="root"):
    """Return exact mismatches and the number of scalar/tensor leaves checked."""
    mismatches, leaves = [], 0
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [f"{path}: keys differ"], 0
        pairs = ((key, left[key], right[key]) for key in left)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: lengths differ"], 0
        pairs = ((i, a, b) for i, (a, b) in enumerate(zip(left, right, strict=True)))
    else:
        if isinstance(left, torch.Tensor):
            equal = left.dtype == right.dtype and torch.equal(left, right)
            detail = ""
            if not equal and left.shape == right.shape:
                detail = f"; max_abs={(left.double() - right.double()).abs().max().item()}"
        else:
            equal, detail = left == right, ""
        return ([] if equal else [f"{path}: values differ{detail}"]), 1
    for key, a, b in pairs:
        differences, count = compare_tree(a, b, f"{path}.{key}")
        mismatches.extend(differences)
        leaves += count
    return mismatches, leaves


def model_state(checkpoint):
    state = {}
    for filename in sorted((checkpoint / "model" / "consolidated").glob("*.safetensors")):
        values = load_file(filename)
        assert not state.keys() & values.keys()
        state.update(values)
    assert state, f"No consolidated model in {checkpoint}"
    return state


def optimizer_state(checkpoint):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "optimizer.pt"
        dcp_to_torch_save(checkpoint / "optim", output)
        return torch.load(output, map_location="cpu", weights_only=False)


def compare(original, resumed, output):
    """Check the second update and each run's exported/checkpoint adapter agreement."""
    report = {"original": str(original), "resumed": str(resumed), "checks": {}}

    def check(name, left, right):
        differences, leaves = compare_tree(left, right)
        report["checks"][name] = {"leaves": leaves, "mismatches": differences}

    a, b = original / "epoch_0_step_1", resumed / "epoch_0_step_1"
    check("model", model_state(a), model_state(b))
    check("optimizer_scheduler", optimizer_state(a), optimizer_state(b))
    check(
        "losses",
        json.loads((a / "losses.json").read_text()),
        json.loads((b / "losses.json").read_text()),
    )
    for filename in ("step_scheduler.pt", "contract.pt"):
        check(
            filename,
            torch.load(a / filename, weights_only=False),
            torch.load(b / filename, weights_only=False),
        )
    for filename in sorted((a / "dataloader").glob("*.pt")):
        check(
            f"dataloader/{filename.name}",
            torch.load(filename, weights_only=False),
            torch.load(b / "dataloader" / filename.name, weights_only=False),
        )
    for run in (original, resumed):
        for checkpoint in sorted(run.glob("epoch_*_step_*")):
            step = int(checkpoint.name.rsplit("_", 1)[1]) + 1
            export = load_file(run / "exports" / f"step-{step:06d}" / "adapter.safetensors")
            saved = {
                key.removeprefix("adapter."): value
                for key, value in model_state(checkpoint).items()
            }
            check(f"{run.name}/{checkpoint.name}/export", export, saved)
    report["exact"] = all(not item["mismatches"] for item in report["checks"].values())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["exact"], "Fresh-process replay or checkpoint/export comparison failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.original, args.resumed, args.output)

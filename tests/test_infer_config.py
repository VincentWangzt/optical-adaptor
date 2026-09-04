from pathlib import Path

from optical_agent.infer_ocr import load_inference_config


def test_inference_yaml_has_qwen3_5_4b() -> None:
    inference, models = load_inference_config(Path("configs/inference.yaml"))
    assert inference.prompt.startswith("Transcribe all visible text")
    assert inference.max_new_tokens == 8192
    assert inference.enable_thinking is False
    assert inference.enforce_eager is True
    assert inference.gpu_memory_utilization == 0.92
    assert inference.trust_remote_code is False
    assert list(models) == ["qwen3.5-4b"]
    assert models["qwen3.5-4b"].model_id == "Qwen/Qwen3.5-4B"
    assert models["qwen3.5-4b"].prompt is None
    assert models["qwen3.5-4b"].max_model_len == 32768


def test_model_specific_configs_load() -> None:
    cases = {
        "inference.qwen3.5-4b.yaml": ("qwen3.5-4b", "qwen", None, None, True),
        "inference.deepseek-ocr.yaml": (
            "deepseek-ocr",
            "deepseek",
            640,
            640,
            False,
        ),
        "inference.deepseek-ocr-2.yaml": (
            "deepseek-ocr-2",
            "deepseek",
            1024,
            768,
            False,
        ),
    }
    for filename, (alias, backend, base_size, image_size, crop_mode) in cases.items():
        _, models = load_inference_config(Path("configs") / filename)
        assert list(models) == [alias]
        assert models[alias].backend == backend
        assert models[alias].base_size == base_size
        assert models[alias].image_size == image_size
        assert models[alias].crop_mode is crop_mode

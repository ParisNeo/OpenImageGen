
"""Tests for hardware-dependent model execution hints."""

from openimagegen.hardware import model_execution_hint


def _hardware(vram=None, ram=64, available=None, device="cuda"):
    return {
        "cpu_logical_count": 16,
        "system_memory_total_gb": ram,
        "system_memory_available_gb": ram if available is None else available,
        "active_device": device,
        "gpus": [] if vram is None else [{
            "index": 0,
            "name": "Test GPU",
            "total_memory_gb": vram,
        }],
    }


def test_small_model_can_be_fully_gpu_resident():
    hint = model_execution_hint(
        "sd15",
        {"name": "example/sd15", "type": "stable_diffusion"},
        _hardware(vram=8, ram=32),
    )
    assert hint["status"] == "gpu"
    assert hint["label"] == "100% GPU"


def test_large_model_recommends_offloading_when_possible():
    hint = model_execution_hint(
        "flux",
        {"name": "black-forest-labs/FLUX.1-schnell", "type": "flux"},
        _hardware(vram=12, ram=64),
    )
    assert hint["status"] == "offload"


def test_model_is_rejected_when_neither_gpu_nor_cpu_can_fit():
    hint = model_execution_hint(
        "qwen",
        {"name": "Qwen/Qwen-Image", "type": "qwen_image"},
        _hardware(vram=4, ram=16),
    )
    assert hint["status"] == "impossible"


def test_remote_models_do_not_depend_on_local_hardware():
    hint = model_execution_hint(
        "hosted",
        {"name": "gpt-image-1", "type": "openai_image"},
        _hardware(vram=None, ram=4, device="cpu"),
    )
    assert hint["status"] == "remote"

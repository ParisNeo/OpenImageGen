
"""
@file        hardware.py
@author      ParisNeo
@description Local hardware detection and conservative model feasibility hints.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch


def _system_memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """Return total and available physical RAM without adding a psutil dependency."""
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total_physical / 1024**3, status.available_physical / 1024**3
        return None, None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES") / 1024**3
        available = page_size * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
        return total, available
    except (AttributeError, OSError, ValueError):
        return None, None


def _rounded(value: Optional[float]) -> Optional[float]:
    return round(value, 1) if value is not None else None


def hardware_profile(device: torch.device) -> Dict[str, Any]:
    """Capture the CPUs, RAM, and GPUs visible to the server process."""
    total_ram, available_ram = _system_memory_gb()
    gpus = []
    if device.type == "cuda" and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": properties.name,
                "total_memory_gb": round(properties.total_memory / 1024**3, 1),
                "unified_memory": False,
            })
    elif device.type == "mps":
        gpus.append({
            "index": 0,
            "name": "Apple Silicon GPU (MPS)",
            "total_memory_gb": _rounded(total_ram),
            "unified_memory": True,
        })

    return {
        "cpu_logical_count": os.cpu_count() or 0,
        "system_memory_total_gb": _rounded(total_ram),
        "system_memory_available_gb": _rounded(available_ram),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "active_device": str(device),
    }


def _memory_requirements(model_key: str, info: Dict[str, Any]) -> Dict[str, float]:
    """Approximate resident memory, smallest offloaded GPU component, and CPU RAM.

    These are deliberately conservative planning estimates, not guarantees. Actual
    usage varies with resolution, batch size, Diffusers version, and backends.
    """
    model_type = info.get("type", "")
    name = str(info.get("name", "")).lower()

    profiles = {
        "stable_diffusion": (4.0, 2.0, 10.0),
        "stable_diffusion_xl": (8.0, 4.0, 20.0),
        "stable_diffusion_3": (18.0, 10.0, 40.0),
        "flux": (24.0, 10.0, 52.0),
        "qwen_image": (56.0, 16.0, 96.0),
        "qwen_image_edit": (56.0, 16.0, 96.0),
        "qwen_image_layered": (56.0, 16.0, 96.0),
        "pixart_sigma": (8.0, 5.0, 20.0),
        "sana": (6.0, 4.0, 14.0),
        "kandinsky": (8.0, 4.0, 20.0),
    }
    resident, offload_vram, cpu_ram = profiles.get(model_type, (8.0, 4.0, 20.0))

    if model_type == "flux_gguf":
        quantization = str(
            info.get("quantization") or info.get("gguf_filename") or model_key
        ).upper().replace("-", "_")
        transformer_gb = 8.0
        for marker, size in (
            ("Q2_K", 4.0),
            ("Q4_K_S", 6.8),
            ("Q4_0", 6.8),
            ("Q5", 8.5),
            ("Q6", 10.5),
            ("Q8_0", 12.7),
        ):
            if marker in quantization:
                transformer_gb = size
                break
        # Quantization covers the transformer only; text encoders and VAE remain.
        resident = transformer_gb + 12.0
        offload_vram = 10.0
        cpu_ram = resident + 12.0

    if model_type in {"stable_diffusion", "stable_diffusion_xl"} and "turbo" in name:
        cpu_ram += 2.0

    return {
        "resident_gb": resident,
        "full_gpu_gb": math.ceil(resident * 1.15),
        "offload_vram_gb": offload_vram,
        "offload_ram_gb": math.ceil(resident + 6.0),
        "cpu_ram_gb": math.ceil(cpu_ram),
    }


def model_execution_hint(
    model_key: str,
    info: Dict[str, Any],
    hardware: Dict[str, Any],
    force_gpu: bool = False,
) -> Dict[str, Any]:
    """Classify whether a model should fit this PC and which execution mode to use."""
    if info.get("type") == "openai_image":
        return {
            "status": "remote",
            "label": "Runs remotely",
            "detail": "No local GPU or model RAM is required; the provider performs generation.",
        }

    requirements = _memory_requirements(model_key, info)
    gpus = hardware.get("gpus", [])
    largest_gpu = max(
        (gpu for gpu in gpus if gpu.get("total_memory_gb") is not None),
        key=lambda gpu: gpu["total_memory_gb"],
        default=None,
    )
    total_ram = hardware.get("system_memory_total_gb")
    available_ram = hardware.get("system_memory_available_gb")
    cpu_count = hardware.get("cpu_logical_count") or 0

    base = {
        "estimated_resident_gb": requirements["resident_gb"],
        "full_gpu_min_gb": requirements["full_gpu_gb"],
        "offload_min_vram_gb": requirements["offload_vram_gb"],
        "cpu_min_ram_gb": requirements["cpu_ram_gb"],
    }

    if largest_gpu and largest_gpu["total_memory_gb"] >= requirements["full_gpu_gb"]:
        detail = (
            f"Estimated {requirements['resident_gb']:g} GB resident; approximately "
            f"{requirements['full_gpu_gb']:g} GB VRAM recommended. "
            f"{largest_gpu['name']} has {largest_gpu['total_memory_gb']:g} GB."
        )
        if len(gpus) > 1:
            detail += " GPUs are not pooled; the largest single GPU determines fit."
        return {**base, "status": "gpu", "label": "100% GPU", "detail": detail}

    can_offload = (
        hardware.get("active_device") == "cuda"
        and largest_gpu is not None
        and largest_gpu["total_memory_gb"] >= requirements["offload_vram_gb"]
        and total_ram is not None
        and total_ram >= requirements["offload_ram_gb"]
    )
    if can_offload:
        detail = (
            f"Use GPU + CPU offload: about {requirements['offload_vram_gb']:g} GB VRAM "
            f"and {requirements['offload_ram_gb']:g} GB system RAM recommended. "
            "Slower than fully GPU-resident execution."
        )
        if available_ram is not None and available_ram < requirements["offload_ram_gb"]:
            detail += f" Only {available_ram:g} GB RAM is currently available; close other applications first."
        return {
            **base,
            "status": "offload",
            "label": "GPU + CPU offload",
            "detail": detail,
        }

    if total_ram is not None and total_ram >= requirements["cpu_ram_gb"]:
        detail = (
            f"CPU generation should fit in approximately {requirements['cpu_ram_gb']:g} GB RAM"
            + (f" using {cpu_count} logical CPUs" if cpu_count else "")
            + ", but it will be very slow."
        )
        if force_gpu:
            detail += " Your current force_gpu=true setting blocks this mode."
        if available_ram is not None and available_ram < requirements["cpu_ram_gb"]:
            detail += f" Currently available RAM is only {available_ram:g} GB."
        return {
            **base,
            "status": "cpu",
            "label": "CPU only · very slow",
            "detail": detail,
        }

    if total_ram is None and largest_gpu is None:
        return {
            **base,
            "status": "unknown",
            "label": "Hardware unknown",
            "detail": (
                f"Plan for roughly {requirements['full_gpu_gb']:g} GB VRAM, or "
                f"{requirements['cpu_ram_gb']:g} GB RAM for very slow CPU generation."
            ),
        }

    ram_text = f"{total_ram:g} GB RAM" if total_ram is not None else "unknown RAM"
    gpu_text = (
        f"largest GPU has {largest_gpu['total_memory_gb']:g} GB VRAM"
        if largest_gpu else "no compatible GPU detected"
    )
    return {
        **base,
        "status": "impossible",
        "label": "Not feasible on this PC",
        "detail": (
            f"Estimated needs: {requirements['full_gpu_gb']:g} GB VRAM for full GPU, "
            f"{requirements['offload_vram_gb']:g} GB VRAM + "
            f"{requirements['offload_ram_gb']:g} GB RAM for offload, or "
            f"{requirements['cpu_ram_gb']:g} GB RAM for CPU. Detected: {gpu_text}; {ram_text}."
        ),
    }

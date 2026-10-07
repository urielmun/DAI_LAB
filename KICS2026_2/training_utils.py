"""Small utilities shared by command-line trainers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, and fail loudly for unavailable explicit CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load full experiment checkpoints across PyTorch 2.x defaults."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def rng_state_payload() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parameter_checksum(module: torch.nn.Module) -> float:
    """A compact diagnostic, not a cryptographic checksum."""

    with torch.no_grad():
        return float(
            sum(parameter.detach().double().sum().item() for parameter in module.parameters())
        )

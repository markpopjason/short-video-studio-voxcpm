from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class StudioConfig:
    model_id: str = os.getenv("VOXCPM_MODEL_ID", "OpenBMB/VoxCPM2")
    model_dir: Path = Path(
        os.getenv("VOXCPM_MODEL_DIR", PROJECT_ROOT / "pretrained_models" / "VoxCPM2")
    )
    output_dir: Path = Path(os.getenv("VOXCPM_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    host: str = os.getenv("VOXCPM_HOST", "127.0.0.1")
    port: int = int(os.getenv("VOXCPM_PORT", "8808"))
    device: str = os.getenv("VOXCPM_DEVICE", "auto")
    prefer_modelscope: bool = _env_bool("VOXCPM_USE_MODELSCOPE", True)
    load_denoiser: bool = _env_bool("VOXCPM_LOAD_DENOISER", False)
    share: bool = _env_bool("VOXCPM_SHARE", False)


DEFAULT_CONFIG = StudioConfig()


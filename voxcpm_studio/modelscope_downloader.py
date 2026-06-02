from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG


_MODEL_MARKERS = (
    "config.json",
    "model.safetensors",
    "model-00001-of-00002.safetensors",
    "model-00001-of-00003.safetensors",
)


def looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _MODEL_MARKERS) or any(
        path.glob("*.safetensors")
    )


def download_model(
    model_id: str = DEFAULT_CONFIG.model_id,
    local_dir: Path | str = DEFAULT_CONFIG.model_dir,
    *,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    target_dir = Path(local_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if looks_like_model_dir(target_dir) and not force:
        print(f"Model already exists: {target_dir}")
        return target_dir

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope is not installed. Run `pip install modelscope` first."
        ) from exc

    kwargs: dict[str, object] = {"local_dir": str(target_dir)}
    if revision:
        kwargs["revision"] = revision

    print(f"Downloading {model_id} from ModelScope to {target_dir} ...")
    snapshot_download(model_id, **kwargs)
    print(f"Model ready: {target_dir}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCPM model from ModelScope")
    parser.add_argument("--model-id", default=DEFAULT_CONFIG.model_id)
    parser.add_argument("--local-dir", default=str(DEFAULT_CONFIG.model_dir))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_model(
        model_id=args.model_id,
        local_dir=args.local_dir,
        revision=args.revision,
        force=args.force,
    )


if __name__ == "__main__":
    main()

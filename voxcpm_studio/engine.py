from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_CONFIG, StudioConfig
from .modelscope_downloader import download_model, looks_like_model_dir


@dataclass
class GenerateResult:
    sample_rate: int
    waveform: Any


class VoxCPMEngine:
    def __init__(
        self,
        config: StudioConfig = DEFAULT_CONFIG,
        *,
        model_id: Optional[str] = None,
        model_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        prefer_modelscope: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.model_id = model_id or config.model_id
        self.model_dir = Path(model_dir or config.model_dir).expanduser().resolve()
        self.device = device or config.device
        self.prefer_modelscope = (
            config.prefer_modelscope if prefer_modelscope is None else prefer_modelscope
        )
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_source(self) -> str:
        if looks_like_model_dir(self.model_dir):
            return str(self.model_dir)
        return str(self.model_dir if self.prefer_modelscope else self.model_id)

    def ensure_local_model(self) -> Path:
        return download_model(self.model_id, self.model_dir)

    def load_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if self.prefer_modelscope:
                source = self.ensure_local_model()
            else:
                source = self.model_dir if looks_like_model_dir(self.model_dir) else self.model_id

            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError(
                    "VoxCPM is not installed. Run `pip install voxcpm` first."
                ) from exc

            kwargs: dict[str, object] = {
                "device": self.device,
                "load_denoiser": self.config.load_denoiser,
            }
            if self.device.startswith("cuda"):
                kwargs["optimize"] = True

            try:
                self._model = VoxCPM.from_pretrained(str(source), **kwargs)
            except TypeError:
                kwargs.pop("load_denoiser", None)
                self._model = VoxCPM.from_pretrained(str(source), **kwargs)

            return self._model

    @staticmethod
    def _clean_control(control: str) -> str:
        control = (control or "").strip()
        return re.sub(r"[()（）]", "", control).strip()

    @classmethod
    def build_text(cls, text: str, control: str = "") -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("请输入要合成的文本。")

        clean_control = cls._clean_control(control)
        return f"({clean_control}){text}" if clean_control else text

    def generate(
        self,
        *,
        text: str,
        reference_wav_path: Optional[str] = None,
        control: str = "",
        prompt_text: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = True,
        denoise: bool = False,
    ) -> GenerateResult:
        model = self.load_model()
        reference = reference_wav_path or None
        clean_prompt_text = (prompt_text or "").strip()
        use_prompt_clone = bool(reference and clean_prompt_text)
        final_text = self.build_text(text, "" if use_prompt_clone else control)

        kwargs: dict[str, object] = {
            "text": final_text,
            "reference_wav_path": reference,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "normalize": bool(normalize),
            "denoise": bool(denoise),
        }
        if use_prompt_clone:
            kwargs["prompt_wav_path"] = reference
            kwargs["prompt_text"] = clean_prompt_text

        waveform = model.generate(**kwargs)
        sample_rate = int(getattr(model.tts_model, "sample_rate", 48000))
        return GenerateResult(sample_rate=sample_rate, waveform=waveform)

    def generate_to_file(
        self,
        output_path: Path | str,
        **kwargs,
    ) -> Path:
        result = self.generate(**kwargs)

        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is not installed. Run `pip install soundfile` first."
            ) from exc

        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), result.waveform, result.sample_rate)
        return output

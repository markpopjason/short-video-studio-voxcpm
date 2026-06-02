from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG
from .modelscope_downloader import download_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxcpm-studio")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start Gradio UI/API")
    serve.add_argument("--host", default=DEFAULT_CONFIG.host)
    serve.add_argument("--port", type=int, default=DEFAULT_CONFIG.port)
    serve.add_argument("--device", default=DEFAULT_CONFIG.device)
    serve.add_argument("--model-id", default=DEFAULT_CONFIG.model_id)
    serve.add_argument("--model-dir", default=str(DEFAULT_CONFIG.model_dir))
    serve.add_argument("--share", action="store_true")
    serve.add_argument("--no-modelscope", action="store_true")

    dl = subparsers.add_parser("download", help="Download model from ModelScope")
    dl.add_argument("--model-id", default=DEFAULT_CONFIG.model_id)
    dl.add_argument("--model-dir", default=str(DEFAULT_CONFIG.model_dir))
    dl.add_argument("--force", action="store_true")

    synth = subparsers.add_parser("synthesize", help="Generate one WAV file")
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", default=str(DEFAULT_CONFIG.output_dir / "out.wav"))
    synth.add_argument("--reference-audio")
    synth.add_argument("--control", default="")
    synth.add_argument("--prompt-text", default="")
    synth.add_argument("--cfg", type=float, default=2.0)
    synth.add_argument("--steps", type=int, default=10)
    synth.add_argument("--device", default=DEFAULT_CONFIG.device)
    synth.add_argument("--model-id", default=DEFAULT_CONFIG.model_id)
    synth.add_argument("--model-dir", default=str(DEFAULT_CONFIG.model_dir))
    synth.add_argument("--no-modelscope", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {None, "serve"}:
        from .server import run_server

        run_server(
            host=getattr(args, "host", DEFAULT_CONFIG.host),
            port=getattr(args, "port", DEFAULT_CONFIG.port),
            device=getattr(args, "device", DEFAULT_CONFIG.device),
            model_id=getattr(args, "model_id", DEFAULT_CONFIG.model_id),
            model_dir=getattr(args, "model_dir", DEFAULT_CONFIG.model_dir),
            prefer_modelscope=not getattr(args, "no_modelscope", False),
            share=getattr(args, "share", False),
        )
        return

    if args.command == "download":
        download_model(args.model_id, args.model_dir, force=args.force)
        return

    if args.command == "synthesize":
        from .engine import VoxCPMEngine

        engine = VoxCPMEngine(
            model_id=args.model_id,
            model_dir=args.model_dir,
            device=args.device,
            prefer_modelscope=not args.no_modelscope,
        )
        output = engine.generate_to_file(
            Path(args.output),
            text=args.text,
            reference_wav_path=args.reference_audio,
            control=args.control,
            prompt_text=args.prompt_text,
            cfg_value=args.cfg,
            inference_timesteps=args.steps,
        )
        print(f"Saved: {output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

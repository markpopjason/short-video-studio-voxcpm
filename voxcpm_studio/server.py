from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gradio as gr

from .config import DEFAULT_CONFIG, StudioConfig
from .engine import VoxCPMEngine


MODES = ["可控克隆", "极致克隆", "声音设计"]


def _mode_components(mode: str):
    needs_ref = mode in {"可控克隆", "极致克隆"}
    needs_prompt = mode == "极致克隆"
    needs_control = mode != "极致克隆"
    return (
        gr.update(visible=needs_ref),
        gr.update(visible=needs_prompt),
        gr.update(visible=needs_control, interactive=needs_control),
    )


def create_interface(engine: VoxCPMEngine) -> gr.Blocks:
    def download_action():
        path = engine.ensure_local_model()
        return f"模型已就绪：{path}"

    def generate_action(
        text: str,
        mode: str,
        reference_wav: Optional[str],
        control: str,
        prompt_text: str,
        cfg_value: float,
        steps: int,
        normalize: bool,
        denoise: bool,
        consent: bool,
    ):
        if not consent:
            raise gr.Error("请先确认你拥有参考音频的使用授权。")
        if mode in {"可控克隆", "极致克隆"} and not reference_wav:
            raise gr.Error("当前模式需要上传参考音频。")
        if mode == "极致克隆" and not (prompt_text or "").strip():
            raise gr.Error("极致克隆需要填写参考音频对应的文本。")

        result = engine.generate(
            text=text,
            reference_wav_path=reference_wav if mode != "声音设计" else None,
            control=control if mode != "极致克隆" else "",
            prompt_text=prompt_text if mode == "极致克隆" else "",
            cfg_value=cfg_value,
            inference_timesteps=int(steps),
            normalize=normalize,
            denoise=denoise,
        )
        return (result.sample_rate, result.waveform), "生成完成"

    css = """
    .compact textarea { font-size: 15px !important; }
    footer { visibility: hidden; }
    """

    with gr.Blocks(
        title="VoxCPM Studio",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:
        gr.Markdown("# VoxCPM Studio")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                text = gr.Textbox(
                    label="目标文本",
                    value="你好，这是一段使用 VoxCPM 生成的声音克隆测试。",
                    lines=4,
                    elem_classes=["compact"],
                )
                mode = gr.Radio(MODES, value="可控克隆", label="模式")
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="参考音频",
                )
                prompt_text = gr.Textbox(
                    label="参考音频文本",
                    placeholder="极致克隆时填写参考音频的原文",
                    lines=3,
                    visible=False,
                )
                control = gr.Textbox(
                    label="风格控制",
                    placeholder="如：温柔、自然、语速稍慢",
                    lines=2,
                )
                consent = gr.Checkbox(
                    label="我确认拥有参考音频的使用授权，并会标注合成语音",
                    value=False,
                )

                with gr.Accordion("参数", open=False):
                    cfg_value = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="CFG")
                    steps = gr.Slider(1, 50, value=10, step=1, label="推理步数")
                    normalize = gr.Checkbox(value=True, label="文本规范化")
                    denoise = gr.Checkbox(value=False, label="参考音频降噪")

                with gr.Row():
                    generate_btn = gr.Button("生成", variant="primary")
                    download_btn = gr.Button("下载/检查模型")

            with gr.Column(scale=4, min_width=320):
                output_audio = gr.Audio(label="生成音频")
                status = gr.Textbox(label="状态", value=f"模型来源：{engine.model_source}")

        mode.change(
            fn=_mode_components,
            inputs=[mode],
            outputs=[reference_wav, prompt_text, control],
            queue=False,
        )
        download_btn.click(fn=download_action, outputs=[status])
        generate_btn.click(
            fn=generate_action,
            inputs=[
                text,
                mode,
                reference_wav,
                control,
                prompt_text,
                cfg_value,
                steps,
                normalize,
                denoise,
                consent,
            ],
            outputs=[output_audio, status],
            api_name="generate_voice",
        )

    return demo


def run_server(
    *,
    host: str = DEFAULT_CONFIG.host,
    port: int = DEFAULT_CONFIG.port,
    device: str = DEFAULT_CONFIG.device,
    model_id: str = DEFAULT_CONFIG.model_id,
    model_dir: Path | str = DEFAULT_CONFIG.model_dir,
    prefer_modelscope: bool = DEFAULT_CONFIG.prefer_modelscope,
    share: bool = DEFAULT_CONFIG.share,
) -> None:
    config = StudioConfig(
        model_id=model_id,
        model_dir=Path(model_dir),
        output_dir=DEFAULT_CONFIG.output_dir,
        host=host,
        port=port,
        device=device,
        prefer_modelscope=prefer_modelscope,
        load_denoiser=DEFAULT_CONFIG.load_denoiser,
        share=share,
    )
    engine = VoxCPMEngine(config)
    interface = create_interface(engine)
    interface.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=True,
        inbrowser=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoxCPM Studio Gradio server")
    parser.add_argument("--host", default=DEFAULT_CONFIG.host)
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG.port)
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument("--model-id", default=DEFAULT_CONFIG.model_id)
    parser.add_argument("--model-dir", default=str(DEFAULT_CONFIG.model_dir))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-modelscope", action="store_true")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        device=args.device,
        model_id=args.model_id,
        model_dir=args.model_dir,
        prefer_modelscope=not args.no_modelscope,
        share=args.share,
    )


if __name__ == "__main__":
    main()


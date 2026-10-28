"""Gradio UI for local text-to-image generation using Diffusers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

from image_agent import PromptPlanningAgent

MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/sd-turbo")
OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
agent = PromptPlanningAgent()


@lru_cache(maxsize=1)
def load_pipeline() -> StableDiffusionPipeline:
    """Load model once and select dtype/device based on available hardware."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True
    )
    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def generate_image(prompt: str, style: str, steps: int, seed: int) -> tuple[str | None, str]:
    """Plan, generate and save one image. Returns an image path and status text."""
    try:
        plan = agent.plan(prompt, style, steps)
        pipeline = load_pipeline()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        result = pipeline(
            prompt=plan.enhanced_prompt,
            negative_prompt=plan.negative_prompt,
            num_inference_steps=plan.steps,
            guidance_scale=plan.guidance_scale,
            generator=generator,
            height=512,
            width=512,
        )
        output_path = OUTPUT_DIR / f"image_{int(seed) % (2**32)}_{os.getpid()}.png"
        result.images[0].save(output_path)
        return str(output_path), (
            f"Image saved to {output_path}\nModel: {MODEL_ID}\nDevice: {device}\n"
            f"Enhanced prompt: {plan.enhanced_prompt}"
        )
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Generation failed ({type(exc).__name__}): {exc}"


with gr.Blocks(title="Free Local Image Generator Agent") as demo:
    gr.Markdown(
        "# Free Local Image Generator Agent\n"
        "Prompt planning + local Diffusers model. No paid image API is used. "
        "First run downloads model weights and may take time."
    )
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Describe your image",
                placeholder="A small glass greenhouse on Mars at sunrise...",
                lines=4,
                max_lines=8,
            )
            style_input = gr.Dropdown(
                choices=["Natural", "Photorealistic", "Digital art", "Watercolor", "Anime", "3D render"],
                value="Natural",
                label="Visual style",
            )
            steps_input = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Inference steps")
            seed_input = gr.Number(value=42, precision=0, label="Seed (reproducibility)")
            generate_button = gr.Button("Generate image", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Generated image", type="filepath")
            status_output = gr.Textbox(label="Agent / runtime details", lines=5)
    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, style_input, steps_input, seed_input],
        outputs=[image_output, status_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )

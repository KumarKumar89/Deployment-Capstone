# Free Local Image Generator Agent

A starter text-to-image application with a deterministic prompt-planning agent and a local Hugging Face Diffusers pipeline. It does **not** call a paid image-generation API.

## Features

- Gradio browser UI.
- Prompt planning for Natural, Photorealistic, Digital art, Watercolor, Anime and 3D render styles.
- Local image generation and PNG output.
- Reproducible seed and bounded inference-step input.
- CPU fallback; CUDA is used when PyTorch detects a compatible GPU.
- Unit tests for the prompt planner, which do not download model weights.

## Requirements and licensing

Python 3.11 is recommended. Internet access is needed for the initial package/model download; afterward, generation can work offline if model files are cached. A compatible GPU is strongly recommended; CPU generation may be slow and memory-intensive.

Python packages listed in `requirements.txt` are free to install. Model weights have their own license and hardware requirements, so review the model card and license before commercial use. Default model: [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo). Change it with `IMAGE_MODEL_ID`.

## Install

Run from this directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation varies by OS and CUDA version. If the default wheel is not appropriate for your machine, install the correct free build from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) first.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:7860`. The first run downloads model weights. Generated PNG files are saved under `outputs/` by default.

### Environment variables

- `IMAGE_MODEL_ID`: Hugging Face model ID (default `stabilityai/sd-turbo`).
- `IMAGE_OUTPUT_DIR`: output directory (default `outputs`).
- `GRADIO_SERVER_NAME`: bind address (default `127.0.0.1`; keep local by default).
- `GRADIO_SERVER_PORT`: port (default `7860`).

Do not bind to `0.0.0.0` or enable public sharing without authentication, request limits and a deployment security review.

## Tests

```bash
python -m pip install pytest
pytest -q
```

Tests cover the prompt-planning agent without loading the image model. They do not prove that the model runs on your machine; validate generation separately after installing a suitable PyTorch build.

## Agent workflow

1. Validate and normalize the user's prompt.
2. Add a style description using transparent, deterministic rules.
3. Bound inference steps to 1–4 and select a low guidance value suitable for SD-Turbo.
4. Pass the plan to Diffusers; the model generates an image.
5. Save the image and report the model, device and enhanced prompt.

This is a lightweight starter agent, not yet a multi-agent system or semantic image-quality evaluator. Future extensions could add an optional local Ollama planner, a separate safety policy, image-quality scoring and a job queue without paid APIs.

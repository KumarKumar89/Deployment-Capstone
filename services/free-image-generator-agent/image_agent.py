"""Small deterministic prompt-planning agent; no paid API or external LLM required."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePlan:
    original_prompt: str
    enhanced_prompt: str
    negative_prompt: str
    steps: int
    guidance_scale: float


class PromptPlanningAgent:
    """Convert a user's description into a bounded, reproducible generation plan."""

    MAX_PROMPT_CHARS = 1_000
    DEFAULT_NEGATIVE = "blurry, low quality, distorted, malformed anatomy, text, watermark"

    def plan(self, prompt: str, style: str = "Natural", steps: int = 2) -> ImagePlan:
        cleaned = " ".join(prompt.split())
        if not cleaned:
            raise ValueError("Please enter an image description.")
        if len(cleaned) > self.MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt must be {self.MAX_PROMPT_CHARS} characters or fewer.")
        styles = {
            "Natural": "natural lighting, coherent composition, detailed",
            "Photorealistic": "photorealistic, realistic materials, natural lighting",
            "Digital art": "polished digital illustration, clear shapes, rich detail",
            "Watercolor": "watercolor painting, paper texture, soft pigment washes",
            "Anime": "anime illustration, expressive shapes, clean linework",
            "3D render": "high-quality 3D render, physically based materials, studio lighting",
        }
        if style not in styles:
            style = "Natural"
        bounded_steps = max(1, min(int(steps), 4))
        return ImagePlan(
            original_prompt=cleaned,
            enhanced_prompt=f"{cleaned}, {styles[style]}",
            negative_prompt=self.DEFAULT_NEGATIVE,
            steps=bounded_steps,
            guidance_scale=0.0,
        )

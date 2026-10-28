import pytest

from image_agent import PromptPlanningAgent


def test_plan_enhances_prompt_and_bounds_steps() -> None:
    plan = PromptPlanningAgent().plan("  a cat   in a garden ", "Watercolor", 99)
    assert plan.original_prompt == "a cat in a garden"
    assert "watercolor painting" in plan.enhanced_prompt
    assert plan.steps == 4
    assert plan.guidance_scale == 0.0


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="image description"):
        PromptPlanningAgent().plan("  ")


def test_oversized_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="characters or fewer"):
        PromptPlanningAgent().plan("x" * 1001)


def test_unknown_style_falls_back_to_natural() -> None:
    plan = PromptPlanningAgent().plan("a mountain", "unknown")
    assert "natural lighting" in plan.enhanced_prompt

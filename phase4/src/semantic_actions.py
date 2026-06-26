"""Prompt-specific semantic action sets for Phase 4."""
from __future__ import annotations

from typing import Any


def prompt_type(prompt: str, action_config: dict[str, Any] | None = None) -> str:
    prompt_value = prompt.lower().strip()
    if action_config:
        aliases = {str(key).lower(): str(value) for key, value in dict(action_config.get("prompt_type_aliases", {})).items()}
        if prompt_value in aliases:
            return aliases[prompt_value]
    if "headphone" in prompt_value:
        return "headphones"
    if "sunglasses" in prompt_value or "glasses" in prompt_value:
        return "glasses"
    if "beard" in prompt_value or "stubble" in prompt_value:
        return "beard"
    if "smile" in prompt_value:
        return "smile"
    return "default"


def actions_for_prompt(prompt: str, action_config: dict[str, Any]) -> list[str]:
    kind = prompt_type(prompt, action_config)
    actions = list(dict(action_config.get("prompt_action_sets", {})).get(kind, []))
    if not actions:
        actions = list(action_config.get("all_actions", []))
    return [str(action) for action in actions]


def decode_action_values(theta, action_names: list[str], *, start_index: int = 1, scale: float = 1.0) -> dict[str, float]:
    import numpy as np

    values: dict[str, float] = {}
    for offset, name in enumerate(action_names):
        if start_index + offset >= len(theta):
            values[name] = 0.0
        else:
            values[name] = float(np.tanh(float(theta[start_index + offset])) * scale)
    return values


def action_parameter_names(action_names: list[str]) -> list[str]:
    return ["scale_logit"] + [f"action_{name}" for name in action_names]


__all__ = ["action_parameter_names", "actions_for_prompt", "decode_action_values", "prompt_type"]


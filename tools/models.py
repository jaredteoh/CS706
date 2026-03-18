"""
shared/models.py
================
Single source of truth for:
  - ModelConfig  : LLM backend settings shared across all stages
  - build_model(): returns a CAMEL ModelFactory instance
  - parse_json() : robust JSON extraction from LLM responses

All three stages import from here — never duplicate model setup code.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

from camel.models import ModelFactory
from camel.types import ModelPlatformType


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    LLM backend settings shared across Stage I, II, and III.

    Ollama (local):
        platform   = ModelPlatformType.OLLAMA
        model_type = "llama3"
        url        = "http://localhost:11434/v1"

    OpenAI (cloud):
        platform   = ModelPlatformType.OPENAI
        model_type = "gpt-4o"
        url        = None  (ignored)
    """
    platform: ModelPlatformType = ModelPlatformType.OLLAMA
    model_type: str = "llama3"
    url: str = "http://localhost:11434/v1"
    temperature: float = 0.2
    token_limit: int = 4096


def build_model(config: ModelConfig):
    """
    Returns a CAMEL-compatible model backend using ModelFactory.
    Compatible with camel-ai 0.2.16.
    """
    kwargs = dict(
        model_platform=config.platform,
        model_type=config.model_type,
        model_config_dict={"temperature": config.temperature},
    )
    if config.platform == ModelPlatformType.OLLAMA:
        kwargs["url"] = config.url
    return ModelFactory.create(**kwargs)


# ---------------------------------------------------------------------------
# JSON parsing utility
# ---------------------------------------------------------------------------

def parse_json(text: str, field: Optional[str] = None,
               default: Any = None) -> Any:
    """
    Robustly extracts JSON from an LLM response.

    Handles:
      - Markdown code fences  (```json ... ```)
      - JSON object embedded in free text
      - Optional field extraction from the parsed object

    Args:
        text:    Raw LLM response string.
        field:   If set, returns parsed[field] instead of the full object.
        default: Returned on any parse failure.
    """
    cleaned = text.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Try direct parse first
    try:
        data = json.loads(cleaned)
        return data.get(field, default) if field else data
    except json.JSONDecodeError:
        pass

    # Fall back: find first {...} block in free text
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start:end])
            return data.get(field, default) if field else data
        except json.JSONDecodeError:
            pass

    return default
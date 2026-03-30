import json
from dataclasses import dataclass
from typing import Any, Optional

from camel.models import ModelFactory
from camel.types import ModelPlatformType


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[ModelPlatformType, str, int]] = {
    # Ollama (local)
    "llama3":                (ModelPlatformType.OLLAMA,     "llama3",                        4096),
    # OpenAI
    "gpt-4o":                (ModelPlatformType.OPENAI,     "gpt-4o",                        32768),
    "o3":                    (ModelPlatformType.OPENAI,     "o3",                            32768),
    # Anthropic
    "claude-3-7-sonnet":     (ModelPlatformType.ANTHROPIC,  "claude-3-7-sonnet-20250219",    32768),
    # Google
    "gemini-2.5-flash":      (ModelPlatformType.GEMINI,     "models/gemini-2.5-flash",       32768),
    # DeepSeek
    "deepseek-v3":           (ModelPlatformType.DEEPSEEK,   "deepseek-chat",                 4096),
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    LLM backend settings shared across Stage I, II, and III.
    Build from a friendly name via model_config_from_name(), or manually.
    """
    platform: ModelPlatformType = ModelPlatformType.OLLAMA
    model_type: str = "llama3"
    url: str = "http://localhost:11434/v1"   # only used for Ollama
    temperature: float = 0.2
    token_limit: int = 4096


def model_config_from_name(name: str, temperature: float = 0.2) -> ModelConfig:
    """
    Build a ModelConfig from a friendly model name (e.g. 'gpt-4o').
    Raises ValueError for unknown names.
    """
    if name not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")
    platform, api_id, token_limit = MODEL_REGISTRY[name]
    return ModelConfig(
        platform=platform,
        model_type=api_id,
        temperature=temperature,
        token_limit=token_limit,
    )


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
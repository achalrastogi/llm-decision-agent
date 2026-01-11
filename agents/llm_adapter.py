"""
Enhanced LLM Adapter
--------------------
Key design goals:
- Explicit system prompt support (prompt engineering)
- Structured / schema-driven outputs
- Minimal LLM usage
- Deterministic core with AI augmentation
- Backward compatibility with existing calls
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# =========================
# Provider imports (safe)
# =========================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


# =========================
# ENUMS & DATA MODELS
# =========================
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 800
    temperature: float = 0.3
    timeout: int = 30


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# =========================
# BASE ADAPTER
# =========================
class BaseLLMAdapter(ABC):
    """
    Base adapter with structured generation support.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"llm_adapter.{config.provider.value}")

    # -------------------------
    # NEW: Structured generation
    # -------------------------
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Canonical, expert-level generation API.
        """

        full_prompt = self._build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )

        return self.generate_response(
            prompt=full_prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

    # -------------------------
    # Backward compatibility
    # -------------------------
    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Legacy interface (kept for compatibility).
        """
        raise NotImplementedError

    # -------------------------
    # Prompt construction
    # -------------------------
    def _build_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[Dict[str, Any]],
    ) -> str:
        prompt = f"""
SYSTEM:
{system_prompt}

USER INPUT:
{user_prompt}
""".strip()

        if output_schema:
            prompt += f"""

OUTPUT FORMAT (STRICT JSON ONLY):
{json.dumps(output_schema, indent=2)}

RULES:
- Respond with VALID JSON only
- Do NOT add explanations or markdown
- Use null for unknown or missing values
- Do NOT infer unstated information
"""
        return prompt

    @abstractmethod
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


# =========================
# OPENAI ADAPTER
# =========================
class OpenAIAdapter(BaseLLMAdapter):

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        if not OpenAI:
            raise ImportError("OpenAI SDK not installed")

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)

        self.cost_per_token = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
        }

    def generate_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.config.timeout,
            )

            content = response.choices[0].message.content
            usage = response.usage

            cost = None
            if usage and self.config.model in self.cost_per_token:
                cost = (
                    usage.prompt_tokens * self.cost_per_token[self.config.model]["input"]
                    + usage.completion_tokens * self.cost_per_token[self.config.model]["output"]
                )

            return LLMResponse(
                content=content,
                provider="openai",
                model=self.config.model,
                tokens_used=usage.total_tokens if usage else None,
                cost_estimate=cost,
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            self.logger.error(f"OpenAI error: {e}")
            raise

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        tokens = len(prompt) // 4
        costs = self.cost_per_token.get(self.config.model)
        if not costs:
            return 0.0
        return tokens * costs["input"] + max_tokens * costs["output"]

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


# =========================
# ANTHROPIC ADAPTER
# =========================
class AnthropicAdapter(BaseLLMAdapter):

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        if not Anthropic:
            raise ImportError("Anthropic SDK not installed")

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = Anthropic(api_key=api_key)

    def generate_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.content[0].text

        return LLMResponse(
            content=content,
            provider="anthropic",
            model=self.config.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        return 0.0

    def is_available(self) -> bool:
        return True


# =========================
# GOOGLE GEMINI ADAPTER
# =========================
class GoogleAdapter(BaseLLMAdapter):

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        if not genai:
            raise ImportError("google-genai SDK not installed")

        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided")

        self.client = genai.Client(api_key=api_key)
        self.model = config.model

    def generate_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return LLMResponse(
            content=response.text,
            provider="google",
            model=self.model,
        )

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        return 0.0

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


# =========================
# MOCK ADAPTER
# =========================
class MockLLMAdapter(BaseLLMAdapter):

    def generate_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "task_type": "analysis",
                    "latency_tolerance": "interactive",
                    "budget_sensitivity": "budget-conscious",
                    "minimum_context_window": 8000,
                    "required_capabilities": ["text-analysis", "summarization"],
                }
            ),
            provider="mock",
            model="mock",
            tokens_used=50,
            cost_estimate=0.0,
            metadata={"mock": True},
        )

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        return 0.0

    def is_available(self) -> bool:
        return True


# =========================
# FACTORY
# =========================
class LLMAdapterFactory:

    @staticmethod
    def create_adapter(config: LLMConfig) -> BaseLLMAdapter:
        if config.provider == LLMProvider.OPENAI:
            return OpenAIAdapter(config)
        if config.provider == LLMProvider.ANTHROPIC:
            return AnthropicAdapter(config)
        if config.provider == LLMProvider.GOOGLE:
            return GoogleAdapter(config)
        if config.provider == LLMProvider.MOCK:
            return MockLLMAdapter(config)

        raise ValueError(f"Unsupported provider: {config.provider}")


# =========================
# MANAGER (FALLBACK READY)
# =========================
class LLMManager:

    def __init__(
        self,
        primary_config: LLMConfig,
        fallback_configs: Optional[List[LLMConfig]] = None,
    ):
        self.primary = LLMAdapterFactory.create_adapter(primary_config)
        self.fallbacks = []

        for cfg in fallback_configs or []:
            try:
                self.fallbacks.append(LLMAdapterFactory.create_adapter(cfg))
            except Exception:
                pass

    def generate(self, **kwargs) -> LLMResponse:
        try:
            if self.primary.is_available():
                return self.primary.generate(**kwargs)
        except Exception:
            pass

        for fb in self.fallbacks:
            try:
                if fb.is_available():
                    return fb.generate(**kwargs)
            except Exception:
                continue

        raise RuntimeError("All LLM adapters failed")

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        return self.primary.estimate_cost(prompt, max_tokens)

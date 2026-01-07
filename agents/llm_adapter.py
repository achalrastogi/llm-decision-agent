"""
Enhanced LLM Adapter with real provider integration
Addresses Critical Gap #1: Integrate actual LLM usage for intelligent agent behavior
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

# Provider imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"  # For testing without API keys


@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"llm_adapter.{config.provider}")
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for the request"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI GPT adapter"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OpenAI:
            raise ImportError("OpenAI package not installed")
        
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        
        # Cost per token (approximate, should be updated regularly)
        self.cost_per_token = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002}
        }
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                timeout=self.config.timeout
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            # Estimate cost
            cost_estimate = None
            if tokens_used and self.config.model in self.cost_per_token:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost_estimate = (
                    input_tokens * self.cost_per_token[self.config.model]["input"] +
                    output_tokens * self.cost_per_token[self.config.model]["output"]
                )
            
            return LLMResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.config.model,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for OpenAI request"""
        if self.config.model not in self.cost_per_token:
            return 0.0
        
        # Rough estimation: 4 chars per token
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = max_tokens
        
        costs = self.cost_per_token[self.config.model]
        return (
            estimated_input_tokens * costs["input"] +
            estimated_output_tokens * costs["output"]
        )
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        try:
            # Simple test call
            self.client.models.list()
            return True
        except Exception:
            return False


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not Anthropic:
            raise ImportError("Anthropic package not installed")
        
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
        
        # Cost per token (approximate)
        self.cost_per_token = {
            "claude-3-opus-20240229": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125}
        }
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Estimate cost
            cost_estimate = None
            if self.config.model in self.cost_per_token:
                costs = self.cost_per_token[self.config.model]
                cost_estimate = (
                    response.usage.input_tokens * costs["input"] +
                    response.usage.output_tokens * costs["output"]
                )
            
            return LLMResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.config.model,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost for Anthropic request"""
        if self.config.model not in self.cost_per_token:
            return 0.0
        
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = max_tokens
        
        costs = self.cost_per_token[self.config.model]
        return (
            estimated_input_tokens * costs["input"] +
            estimated_output_tokens * costs["output"]
        )
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        try:
            # Simple test - this might need adjustment based on Anthropic's API
            return True  # Placeholder - implement actual availability check
        except Exception:
            return False


class GoogleAdapter(BaseLLMAdapter):
    """Google Gemini adapter (google-genai SDK)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed")

        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided")

        # NEW: client-based API
        self.client = genai.Client(api_key=api_key)

        self.model = config.model

    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            content = response.text

            return LLMResponse(
                content=content,
                provider=self.config.provider.value,
                model=self.model,
                tokens_used=None,
                cost_estimate=None,
                metadata={"provider": "google-gemini"}
            )

        except Exception as e:
            self.logger.error(f"Google Gemini API error: {e}")
            raise

    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        # Gemini free tier – cost estimation optional
        return 0.0

    def is_available(self) -> bool:
        try:
            # Lightweight ping
            self.client.models.list()
            return True
        except Exception:
            return False

class MockLLMAdapter(BaseLLMAdapter):
    """Mock adapter for testing without API keys"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.responses = {
            "task_type": "Based on the description, this appears to be an analytical task requiring data processing and insight generation.",
            "constraints": "The user requires: budget-conscious solution, interactive latency, moderate context window (8000 tokens), text generation and analysis capabilities.",
            "recommendation": "I recommend GPT-4 for this use case due to its strong reasoning capabilities, which align with your analytical requirements. While it has premium pricing, the superior accuracy justifies the cost for complex analysis tasks."
        }
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response"""
        # Simple keyword-based response selection
        prompt_lower = prompt.lower()
        
        if "task type" in prompt_lower or "classify" in prompt_lower:
            content = self.responses["task_type"]
        elif "constraint" in prompt_lower or "requirement" in prompt_lower:
            content = self.responses["constraints"]
        elif "recommend" in prompt_lower:
            content = self.responses["recommendation"]
        else:
            content = "This is a mock response for testing purposes. The actual LLM integration would provide intelligent analysis here."
        
        return LLMResponse(
            content=content,
            provider="mock",
            model=self.config.model,
            tokens_used=len(content) // 4,
            cost_estimate=0.001,  # Mock cost
            metadata={"mock": True}
        )
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Mock cost estimation"""
        return 0.001
    
    def is_available(self) -> bool:
        """Mock is always available"""
        return True


class LLMAdapterFactory:
    """Factory for creating LLM adapters"""
    
    @staticmethod
    def create_adapter(config: LLMConfig) -> BaseLLMAdapter:
        """Create appropriate adapter based on provider"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIAdapter(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicAdapter(config)
        elif config.provider == LLMProvider.GOOGLE:
            return GoogleAdapter(config)
        elif config.provider == LLMProvider.MOCK:
            return MockLLMAdapter(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    @staticmethod
    def get_available_providers() -> List[LLMProvider]:
        """Get list of available providers based on installed packages and API keys"""
        available = [LLMProvider.MOCK]  # Mock is always available
        
        # Check OpenAI
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            available.append(LLMProvider.OPENAI)
        
        # Check Anthropic
        if Anthropic and os.getenv("ANTHROPIC_API_KEY"):
            available.append(LLMProvider.ANTHROPIC)
        
        # Check Google
        if genai and os.getenv("GOOGLE_API_KEY"):
            available.append(LLMProvider.GOOGLE)
        
        return available


class LLMManager:
    """Manages multiple LLM adapters with fallback and load balancing"""
    
    def __init__(self, primary_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.primary_adapter = LLMAdapterFactory.create_adapter(primary_config)
        self.fallback_adapters = []
        
        if fallback_configs:
            for config in fallback_configs:
                try:
                    adapter = LLMAdapterFactory.create_adapter(config)
                    self.fallback_adapters.append(adapter)
                except Exception as e:
                    logging.warning(f"Failed to create fallback adapter for {config.provider}: {e}")
        
        self.logger = logging.getLogger("llm_manager")
    
    def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with fallback support"""
        # Try primary adapter first
        try:
            if self.primary_adapter.is_available():
                return self.primary_adapter.generate_response(prompt, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary adapter failed: {e}")
        
        # Try fallback adapters
        for adapter in self.fallback_adapters:
            try:
                if adapter.is_available():
                    self.logger.info(f"Using fallback adapter: {adapter.config.provider}")
                    return adapter.generate_response(prompt, **kwargs)
            except Exception as e:
                self.logger.warning(f"Fallback adapter {adapter.config.provider} failed: {e}")
        
        # If all fail, raise error
        raise RuntimeError("All LLM adapters failed")
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate cost using primary adapter"""
        return self.primary_adapter.estimate_cost(prompt, max_tokens)
    
    def get_status(self) -> Dict[str, bool]:
        """Get availability status of all adapters"""
        status = {"primary": self.primary_adapter.is_available()}
        
        for i, adapter in enumerate(self.fallback_adapters):
            status[f"fallback_{i}"] = adapter.is_available()
        
        return status
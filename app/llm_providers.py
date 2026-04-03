"""
Multi-LLM Provider System
Supports Claude, OpenAI GPT-4, Google Gemini, and Groq with automatic fallback
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze(self, transcript: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Analyze transcript and return structured response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available"""
        pass
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling code blocks"""
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Find start and end of code block
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start_idx:end_idx])
        
        return json.loads(text)


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.settings.anthropic_api_key)
    
    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, transcript: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        
        response = client.messages.create(
            model=self.settings.claude_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt.format(transcript=transcript)}]
        )
        
        return self.parse_json_response(response.content[0].text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-4 provider"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.settings.openai_api_key)
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, transcript: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(transcript=transcript)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return self.parse_json_response(response.choices[0].message.content)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.settings.gemini_api_key)
    
    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.settings.gemini_api_key)
            self._client = genai.GenerativeModel(
                self.settings.gemini_model,
                generation_config={"response_mime_type": "application/json"}
            )
        return self._client
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, transcript: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        
        full_prompt = f"{system_prompt}\n\n{prompt.format(transcript=transcript)}"
        response = client.generate_content(full_prompt)
        
        return self.parse_json_response(response.text)


class GroqProvider(BaseLLMProvider):
    """Groq provider (fast inference)"""
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    def is_available(self) -> bool:
        return bool(self.settings.groq_api_key)
    
    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.settings.groq_api_key)
        return self._client
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, transcript: str, prompt: str, system_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(transcript=transcript)}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return self.parse_json_response(response.choices[0].message.content)


class LLMProviderFactory:
    """Factory for creating and managing LLM providers with fallback support"""
    
    _providers: Dict[LLMProvider, BaseLLMProvider] = {}
    
    @classmethod
    def get_provider(cls, provider_type: LLMProvider) -> BaseLLMProvider:
        """Get or create a provider instance"""
        if provider_type not in cls._providers:
            if provider_type == LLMProvider.CLAUDE:
                cls._providers[provider_type] = ClaudeProvider()
            elif provider_type == LLMProvider.OPENAI:
                cls._providers[provider_type] = OpenAIProvider()
            elif provider_type == LLMProvider.GEMINI:
                cls._providers[provider_type] = GeminiProvider()
            elif provider_type == LLMProvider.GROQ:
                cls._providers[provider_type] = GroqProvider()
            else:
                raise ValueError(f"Unknown provider: {provider_type}")
        
        return cls._providers[provider_type]
    
    @classmethod
    def get_available_providers(cls) -> List[Tuple[LLMProvider, BaseLLMProvider]]:
        """Get all available (configured) providers"""
        settings = get_settings()
        available = []
        
        # Check primary provider first
        primary = cls.get_provider(settings.llm_provider)
        if primary.is_available():
            available.append((settings.llm_provider, primary))
        
        # Add fallback providers
        for provider_type in settings.get_fallback_providers():
            if provider_type != settings.llm_provider:
                provider = cls.get_provider(provider_type)
                if provider.is_available():
                    available.append((provider_type, provider))
        
        return available


class MultiLLMAnalyzer:
    """
    Analyzer that uses multiple LLM providers with automatic fallback
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def analyze_with_fallback(
        self, 
        transcript: str, 
        prompt: str, 
        system_prompt: str
    ) -> Tuple[Dict[str, Any], LLMProvider]:
        """
        Analyze transcript using primary provider with fallback to alternatives.
        Returns: (analysis_result, provider_used)
        """
        available_providers = LLMProviderFactory.get_available_providers()
        
        if not available_providers:
            raise RuntimeError("No LLM providers are configured. Please set at least one API key.")
        
        last_error = None
        
        for provider_type, provider in available_providers:
            try:
                logger.info(f"Attempting analysis with {provider_type.value}")
                result = provider.analyze(transcript, prompt, system_prompt)
                logger.info(f"Analysis successful with {provider_type.value}")
                return result, provider_type
                
            except Exception as e:
                logger.warning(f"Provider {provider_type.value} failed: {e}")
                last_error = e
                
                if not self.settings.llm_enable_fallback:
                    raise
                
                continue
        
        # All providers failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# Singleton instance
_multi_llm_analyzer: Optional[MultiLLMAnalyzer] = None


def get_multi_llm_analyzer() -> MultiLLMAnalyzer:
    """Get or create MultiLLMAnalyzer instance"""
    global _multi_llm_analyzer
    if _multi_llm_analyzer is None:
        _multi_llm_analyzer = MultiLLMAnalyzer()
    return _multi_llm_analyzer

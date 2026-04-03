"""
Configuration settings for the Call Center Compliance API
Supports multiple LLM providers with fallback capabilities
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional, List
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class STTProvider(str, Enum):
    """Supported Speech-to-Text providers"""
    WHISPER_LOCAL = "whisper_local"
    WHISPER_API = "whisper_api"
    GROQ_WHISPER = "groq_whisper"
    GEMINI = "gemini"


class Settings(BaseSettings):
    # ============= API Configuration =============
    app_name: str = "Call Center Compliance API"
    app_env: str = "development"
    debug: bool = True
    api_key: str = ""
    
    # ============= Redis Configuration =============
    redis_url: str = "redis://localhost:6379/0"
    
    # ============= LLM Provider Selection =============
    # Primary LLM provider (claude, openai, gemini, groq)
    llm_provider: LLMProvider = LLMProvider.CLAUDE
    # Fallback providers in order of preference
    llm_fallback_providers: str = "openai,gemini,groq"  # Comma-separated
    # Enable automatic fallback on failure
    llm_enable_fallback: bool = True
    
    # ============= Anthropic (Claude) Configuration =============
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    
    # ============= OpenAI Configuration =============
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    use_openai_whisper_api: bool = False
    
    # ============= Google Gemini Configuration =============
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-pro"
    
    # ============= Groq Configuration =============
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_whisper_model: str = "whisper-large-v3"
    
    # ============= STT Configuration =============
    stt_provider: STTProvider = STTProvider.WHISPER_API
    whisper_model: str = "large-v3"
    
    # ============= Feature Flags =============
    enable_speaker_diarization: bool = True
    enable_pii_detection: bool = True
    enable_pii_redaction: bool = False  # Only redact if explicitly enabled
    enable_sentiment_timeline: bool = True
    enable_call_quality_metrics: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours
    
    # ============= Webhook Configuration =============
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""
    webhook_retry_count: int = 3
    
    # ============= Batch Processing =============
    max_batch_size: int = 10
    batch_timeout_seconds: int = 1800  # 30 minutes
    
    # ============= Rate Limiting =============
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    
    # ============= Custom SOP Templates =============
    default_sop_template: str = "standard"  # standard, banking, telecom, insurance
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_fallback_providers(self) -> List[LLMProvider]:
        """Parse fallback providers from comma-separated string"""
        if not self.llm_fallback_providers:
            return []
        providers = []
        for p in self.llm_fallback_providers.split(","):
            try:
                providers.append(LLMProvider(p.strip().lower()))
            except ValueError:
                continue
        return providers


@lru_cache
def get_settings() -> Settings:
    return Settings()
 
 
 

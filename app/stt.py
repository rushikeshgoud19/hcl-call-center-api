"""
Speech-to-Text processing using OpenAI Whisper
Optimized for Hinglish (Hindi-English) and Tanglish (Tamil-English) code-switching
"""
import os
import logging
from typing import Tuple, Optional
from app.config import get_settings, STTProvider

logger = logging.getLogger(__name__)


# Priming prompt for code-switching (Hinglish/Tanglish)
# This helps Whisper expect mixed-language content
MULTILINGUAL_PROMPT = """
Ji sir, payment ho gaya, I will pay via UPI. Aap ka EMI pending hai.
Hello sir, main ABC company se call kar raha hoon. 
Vanakkam, ungaludaya payment pending irukku. Please pay pannunga.
Partial payment kar sakte hain. Down payment dena hoga.
"""


class SpeechToText:
    """
    Speech-to-Text processor supporting local Whisper or OpenAI Whisper API
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        # Use stt_provider setting to determine which method to use
        self._stt_provider = self.settings.stt_provider
        # Also respect legacy use_openai_whisper_api flag
        self._use_api = (
            self._stt_provider == STTProvider.WHISPER_API or 
            self.settings.use_openai_whisper_api
        )
    
    def _load_local_model(self):
        """Lazy load the local Whisper model"""
        if self._model is None and not self._use_api:
            import whisper
            model_name = self.settings.whisper_model
            logger.info(f"Loading Whisper model: {model_name}")
            self._model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        return self._model
    
    def transcribe_local(self, audio_path: str) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio using local Whisper model.
        Returns: (transcript, error_message)
        """
        try:
            model = self._load_local_model()
            
            # Transcribe with optimizations for multilingual content
            result = model.transcribe(
                audio_path,
                language=None,  # Auto-detect but Whisper handles Hindi/Tamil well
                task="transcribe",
                initial_prompt=MULTILINGUAL_PROMPT,
                word_timestamps=True,
                verbose=False
            )
            
            transcript = result.get("text", "").strip()
            
            if not transcript:
                return "", "No speech detected in audio"
            
            detected_language = result.get("language", "unknown")
            logger.info(f"Transcription complete. Detected language: {detected_language}")
            
            return transcript, None
            
        except Exception as e:
            logger.exception("Error in local Whisper transcription")
            return "", f"Transcription error: {str(e)}"
    
    def transcribe_api(self, audio_path: str) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio using OpenAI Whisper API.
        Returns: (transcript, error_message)
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.settings.openai_api_key)
            
            with open(audio_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    prompt=MULTILINGUAL_PROMPT,
                    response_format="text"
                )
            
            transcript = transcript_response.strip()
            
            if not transcript:
                return "", "No speech detected in audio"
            
            logger.info("API transcription complete")
            return transcript, None
            
        except Exception as e:
            logger.exception("Error in OpenAI Whisper API transcription")
            return "", f"API transcription error: {str(e)}"
    
    def transcribe_groq(self, audio_path: str) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio using Groq Whisper API.
        Returns: (transcript, error_message)
        """
        try:
            if not self.settings.groq_api_key or "your-groq-key-here" in self.settings.groq_api_key:
                return "", "Groq API key not configured properly (still using placeholder)."
                
            from groq import Groq
            client = Groq(api_key=self.settings.groq_api_key)
            
            with open(audio_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model=self.settings.groq_whisper_model,
                    file=("audio.mp3", audio_file.read()),
                    prompt=MULTILINGUAL_PROMPT,
                    response_format="text"
                )
            
            transcript = transcript_response.strip()
            if not transcript:
                return "", "No speech detected in audio"
                
            logger.info("Groq API transcription complete")
            return transcript, None
            
        except ImportError:
            return "", "Groq package not installed. Run `pip install groq`"
        except Exception as e:
            logger.exception("Error in Groq Whisper API transcription")
            return "", f"Groq API transcription error: {str(e)}"

    def transcribe(self, audio_path: str) -> Tuple[str, Optional[str]]:
        """
        Main transcription method. Uses API or local based on configuration.
        Returns: (transcript, error_message)
        """
        if self._stt_provider == STTProvider.GROQ_WHISPER:
            return self.transcribe_groq(audio_path)
        elif self._use_api:
            return self.transcribe_api(audio_path)
        else:
            return self.transcribe_local(audio_path)


# Singleton instance
_stt_instance: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    """Get or create Speech-to-Text instance"""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = SpeechToText()
    return _stt_instance

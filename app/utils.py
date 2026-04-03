"""
Utility functions for audio downloading, conversion, and processing
"""
import os
import httpx
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


async def download_audio(url: str, timeout: int = 120) -> Tuple[str, Optional[str]]:
    """
    Download audio file from URL to a temporary location.
    Returns: (file_path, error_message)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Determine file extension from content-type or URL
            content_type = response.headers.get("content-type", "")
            if "mp3" in content_type or url.endswith(".mp3"):
                ext = ".mp3"
            elif "wav" in content_type or url.endswith(".wav"):
                ext = ".wav"
            elif "m4a" in content_type or url.endswith(".m4a"):
                ext = ".m4a"
            elif "ogg" in content_type or url.endswith(".ogg"):
                ext = ".ogg"
            else:
                ext = ".mp3"  # Default to mp3
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded audio to {temp_path}, size: {len(response.content)} bytes")
            return temp_path, None
            
    except httpx.TimeoutException:
        return "", "Audio download timed out"
    except httpx.HTTPStatusError as e:
        return "", f"HTTP error: {e.response.status_code}"
    except Exception as e:
        logger.exception("Error downloading audio")
        return "", f"Download error: {str(e)}"


def cleanup_temp_file(file_path: str) -> None:
    """Safely remove temporary file"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the audio file is valid and not empty/corrupted.
    Returns: (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(file_path)
        if file_size < 1000:  # Less than 1KB is likely invalid
            return False, "low_audio_quality"
        
        # Additional validation could be done here with pydub
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_audio_duration(file_path: str) -> Optional[int]:
    """Get audio duration in seconds using pydub"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return int(len(audio) / 1000)  # Convert ms to seconds
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return None

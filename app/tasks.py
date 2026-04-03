"""
Celery tasks for async audio processing
Enhanced with webhook notifications, caching, and multi-LLM support
"""
import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Celery
import httpx
from app.config import get_settings, LLMProvider
from app.utils import download_audio, cleanup_temp_file, validate_audio_file, get_audio_duration
from app.stt import get_stt
from app.processor import get_enhanced_processor
from app.models import TaskStatus, SOPTemplate

logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize Celery
celery_app = Celery(
    "call_center_compliance",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540,  # 9 minutes soft limit
    result_expires=86400,  # Results expire after 24 hours
)


# ============= Storage Functions =============

def get_redis_client():
    """Get Redis client for caching and storage"""
    from redis import Redis
    return Redis.from_url(settings.redis_url)


def store_result(task_id: str, result: Dict[str, Any]) -> None:
    """Store task result in Redis"""
    try:
        redis_client = get_redis_client()
        redis_client.setex(
            f"task_result:{task_id}",
            settings.cache_ttl_seconds,
            json.dumps(result, default=str)
        )
        logger.info(f"Stored result for task {task_id}")
    except Exception as e:
        logger.warning(f"Failed to store result in Redis: {e}")


def get_result(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task result from Redis"""
    try:
        redis_client = get_redis_client()
        data = redis_client.get(f"task_result:{task_id}")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to get result from Redis: {e}")
    return None


def get_cached_analysis(audio_url: str) -> Optional[Dict[str, Any]]:
    """Check if we have cached analysis for this audio URL"""
    if not settings.enable_caching:
        return None
    
    try:
        cache_key = f"audio_cache:{hashlib.md5(audio_url.encode()).hexdigest()}"
        redis_client = get_redis_client()
        data = redis_client.get(cache_key)
        if data:
            logger.info(f"Cache hit for audio URL: {audio_url[:50]}...")
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
    return None


def cache_analysis(audio_url: str, result: Dict[str, Any]) -> None:
    """Cache analysis result for audio URL"""
    if not settings.enable_caching:
        return
    
    try:
        cache_key = f"audio_cache:{hashlib.md5(audio_url.encode()).hexdigest()}"
        redis_client = get_redis_client()
        redis_client.setex(
            cache_key,
            settings.cache_ttl_seconds,
            json.dumps(result, default=str)
        )
        logger.info(f"Cached analysis for audio URL: {audio_url[:50]}...")
    except Exception as e:
        logger.warning(f"Cache store failed: {e}")


# ============= Webhook Functions =============

def send_webhook(webhook_url: str, payload: Dict[str, Any], secret: str = "") -> bool:
    """Send webhook notification"""
    try:
        headers = {"Content-Type": "application/json"}
        if secret:
            # Add HMAC signature for webhook verification
            import hmac
            signature = hmac.new(
                secret.encode(),
                json.dumps(payload, default=str).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        response = httpx.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Webhook sent successfully to {webhook_url}")
        return True
        
    except Exception as e:
        logger.error(f"Webhook failed: {e}")
        return False


@celery_app.task(bind=True, name="send_webhook_notification")
def send_webhook_task(self, webhook_url: str, payload: Dict[str, Any]) -> bool:
    """Celery task for sending webhook with retries"""
    for attempt in range(settings.webhook_retry_count):
        if send_webhook(webhook_url, payload, settings.webhook_secret):
            return True
        logger.warning(f"Webhook attempt {attempt + 1} failed, retrying...")
    return False


# ============= Main Analysis Task =============

@celery_app.task(bind=True, name="analyze_audio")
def analyze_audio_task(
    self, 
    audio_url: str,
    sop_template: str = "standard",
    custom_checkpoints: Optional[list] = None,
    enable_pii_redaction: bool = False,
    webhook_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main Celery task for processing audio.
    Pipeline: Cache Check -> Download -> Validate -> STT -> LLM Analysis -> Store -> Webhook
    """
    task_id = self.request.id
    start_time = datetime.utcnow()
    logger.info(f"Starting analysis task {task_id} for URL: {audio_url[:50]}...")
    
    # Convert sop_template string to enum
    try:
        sop_template_enum = SOPTemplate(sop_template)
    except ValueError:
        sop_template_enum = SOPTemplate.STANDARD
    
    # Initialize result structure
    result = {
        "task_id": task_id,
        "status": TaskStatus.PROCESSING.value,
        "created_at": start_time.isoformat(),
        "completed_at": None,
        "processing_time_seconds": None,
        "llm_provider_used": None,
        "transcript": None,
        "transcript_redacted": None,
        "summary": None,
        "diarized_transcript": None,
        "sop_validation": None,
        "analytics": None,
        "keywords": [],
        "keywords_analysis": None,
        "metadata": metadata,
        "error": None,
        "error_code": None,
        "overall_confidence": None
    }
    
    # Check cache first
    cached = get_cached_analysis(audio_url)
    if cached:
        cached["task_id"] = task_id
        cached["metadata"] = metadata
        store_result(task_id, cached)
        
        if webhook_url:
            send_webhook_task.delay(webhook_url, {
                "event": "analysis_complete",
                "task_id": task_id,
                "status": "completed",
                "cached": True
            })
        
        return cached
    
    audio_path = None
    
    try:
        # Step 1: Download audio
        logger.info(f"[{task_id}] Downloading audio...")
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_path, error = loop.run_until_complete(download_audio(audio_url))
        loop.close()
        
        if error:
            raise ValueError(f"Download failed: {error}")
        
        # Step 2: Validate audio
        logger.info(f"[{task_id}] Validating audio...")
        is_valid, validation_error = validate_audio_file(audio_path)
        if not is_valid:
            result["status"] = TaskStatus.FAILED.value
            result["error"] = validation_error
            result["error_code"] = "AUDIO_VALIDATION_FAILED"
            store_result(task_id, result)
            return result
        
        # Get audio duration
        duration = get_audio_duration(audio_path)
        
        # Step 3: Speech-to-Text
        logger.info(f"[{task_id}] Running speech-to-text...")
        stt = get_stt()
        transcript, stt_error = stt.transcribe(audio_path)
        
        if stt_error:
            result["status"] = TaskStatus.FAILED.value
            result["error"] = stt_error
            result["error_code"] = "STT_FAILED"
            store_result(task_id, result)
            return result
        
        result["transcript"] = transcript
        logger.info(f"[{task_id}] Transcription complete: {len(transcript)} chars")
        
        # Step 4: LLM Analysis with multi-provider support
        logger.info(f"[{task_id}] Running LLM analysis...")
        processor = get_enhanced_processor()
        analysis, provider_used = processor.analyze_transcript(
            transcript=transcript,
            sop_template=sop_template_enum,
            custom_checkpoints=custom_checkpoints
        )
        
        result["llm_provider_used"] = provider_used.value
        
        # Parse and store results
        result["summary"] = analysis.get("summary", "")
        result["overall_confidence"] = analysis.get("confidence_score", 0.9)
        
        # Parse SOP validation
        sop_validation = processor.parse_sop_validation(analysis, sop_template_enum)
        result["sop_validation"] = sop_validation.model_dump()
        
        # Parse complete analytics
        analytics = processor.parse_analytics(analysis, transcript, enable_pii_redaction)
        if duration:
            analytics.call_duration_seconds = duration
        result["analytics"] = analytics.model_dump()
        
        # Handle PII redaction
        if enable_pii_redaction and analytics.pii_analysis:
            result["transcript_redacted"] = analytics.pii_analysis.redacted_transcript
        
        # Parse keywords
        keywords_analysis = processor.parse_keywords_analysis(analysis)
        result["keywords"] = keywords_analysis.all_keywords
        result["keywords_analysis"] = keywords_analysis.model_dump()
        
        # Parse diarization
        diarization = processor.parse_diarization(analysis)
        if diarization:
            result["diarized_transcript"] = diarization.model_dump()
        
        # Mark as completed
        end_time = datetime.utcnow()
        result["status"] = TaskStatus.COMPLETED.value
        result["completed_at"] = end_time.isoformat()
        result["processing_time_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"[{task_id}] Analysis completed successfully in {result['processing_time_seconds']:.2f}s")
        
    except Exception as e:
        logger.exception(f"[{task_id}] Task failed with error")
        result["status"] = TaskStatus.FAILED.value
        result["error"] = str(e)
        result["error_code"] = "PROCESSING_ERROR"
    
    finally:
        # Cleanup temporary file
        if audio_path:
            cleanup_temp_file(audio_path)
    
    # Store result
    store_result(task_id, result)
    
    # Cache successful results
    if result["status"] == TaskStatus.COMPLETED.value:
        cache_analysis(audio_url, result)
    
    # Send webhook notification
    if webhook_url:
        send_webhook_task.delay(webhook_url, {
            "event": "analysis_complete",
            "task_id": task_id,
            "status": result["status"],
            "processing_time_seconds": result.get("processing_time_seconds"),
            "error": result.get("error")
        })
    
    return result


# ============= Batch Processing Task =============

@celery_app.task(bind=True, name="analyze_batch")
def analyze_batch_task(
    self,
    audio_urls: list,
    sop_template: str = "standard",
    enable_pii_redaction: bool = False,
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process multiple audio files in batch
    """
    batch_id = self.request.id
    logger.info(f"Starting batch analysis {batch_id} with {len(audio_urls)} files")
    
    task_ids = []
    for url in audio_urls[:settings.max_batch_size]:
        task = analyze_audio_task.delay(
            audio_url=url,
            sop_template=sop_template,
            enable_pii_redaction=enable_pii_redaction
        )
        task_ids.append(task.id)
    
    result = {
        "batch_id": batch_id,
        "task_ids": task_ids,
        "status": "processing",
        "total_files": len(task_ids)
    }
    
    # Store batch info
    try:
        redis_client = get_redis_client()
        redis_client.setex(
            f"batch:{batch_id}",
            settings.cache_ttl_seconds,
            json.dumps(result)
        )
    except Exception as e:
        logger.warning(f"Failed to store batch info: {e}")
    
    # Send webhook if provided
    if webhook_url:
        send_webhook_task.delay(webhook_url, {
            "event": "batch_submitted",
            "batch_id": batch_id,
            "task_ids": task_ids
        })
    
    return result

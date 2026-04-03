"""
Call Center Compliance API - Main FastAPI Application
Enhanced with batch processing, webhooks, and provider selection
"""
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.config import get_settings, Settings, LLMProvider
from app.models import (
    AnalyzeRequest, AnalyzeResponse, TaskResultResponse, TaskStatus,
    BatchAnalyzeRequest, BatchAnalyzeResponse, SOPTemplate
)
import uuid
import time
# Import tasks with fallback for when Celery isn't running
try:
    from app.tasks import analyze_audio_task, analyze_batch_task, get_result
    CELERY_AVAILABLE = True
except Exception:
    CELERY_AVAILABLE = False
    analyze_audio_task = None
    analyze_batch_task = None
    def get_result(task_id):
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting Call Center Compliance API v2.0...")
    logger.info(f"Primary LLM Provider: {get_settings().llm_provider.value}")
    yield
    logger.info("Shutting down Call Center Compliance API...")


# Initialize FastAPI
app = FastAPI(
    title="Call Center Compliance API",
    description="""
    ## AI-Powered Call Center Analytics System
    
    Processes voice recordings in **Hindi (Hinglish)**, **Tamil (Tanglish)**, **Telugu**, and **English**.
    
    ### Key Features
    
    - **Multi-LLM Support**: Claude, GPT-4, Gemini, Groq with automatic fallback
    - **Voice-to-Text**: Whisper-powered transcription with code-switching support
    - **Speaker Diarization**: Identify Agent vs Customer with talk ratios
    - **Sentiment Timeline**: Track sentiment changes throughout the call
    - **SOP Validation**: Validate against multiple industry templates
    - **Payment Categorization**: EMI, Full, Partial, Down Payment detection
    - **PII Detection & Redaction**: Protect sensitive information
    - **Rejection Analysis**: Categorize and understand objections
    - **Batch Processing**: Analyze multiple calls simultaneously
    - **Webhook Notifications**: Get notified when analysis completes
    - **Smart Caching**: Faster responses for repeated requests
    
    ### SOP Templates Available
    - **Standard**: Generic call center compliance
    - **Banking**: Financial services specific
    - **Telecom**: Telecom industry specific
    - **Insurance**: Insurance compliance
    - **Healthcare**: HIPAA-aware healthcare
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for Web UI
import os
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ============= Dependencies =============

def get_settings_dep() -> Settings:
    return get_settings()


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings_dep)
) -> bool:
    """Verify API key if configured"""
    if settings.api_key and settings.api_key != "":
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
    return True


# ============= Health & Info Endpoints =============

@app.get("/", tags=["Health"], include_in_schema=False)
async def root():
    """Redirect to Web UI"""
    return RedirectResponse(url="/ui/")


@app.get("/api", tags=["Health"])
async def api_info():
    """API info and health check"""
    settings = get_settings()
    return {
        "service": "Call Center Compliance API",
        "status": "healthy",
        "version": "2.0.0",
        "primary_llm": settings.llm_provider.value,
        "celery_available": CELERY_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
        "ui_url": "/ui/",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with component status"""
    settings = get_settings()
    
    # Check which providers are configured
    providers_status = {
        "claude": bool(settings.anthropic_api_key),
        "openai": bool(settings.openai_api_key),
        "gemini": bool(settings.gemini_api_key),
        "groq": bool(settings.groq_api_key)
    }
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "api": "up",
            "celery": "up",
            "redis": "up"
        },
        "llm_providers": providers_status,
        "primary_provider": settings.llm_provider.value,
        "features": {
            "speaker_diarization": settings.enable_speaker_diarization,
            "pii_detection": settings.enable_pii_detection,
            "sentiment_timeline": settings.enable_sentiment_timeline,
            "caching": settings.enable_caching,
            "webhooks": settings.webhook_enabled
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/providers", tags=["Info"])
async def list_providers():
    """List available LLM providers and their status"""
    settings = get_settings()
    
    providers = []
    for provider in LLMProvider:
        if provider == LLMProvider.CLAUDE:
            configured = bool(settings.anthropic_api_key)
            model = settings.claude_model
        elif provider == LLMProvider.OPENAI:
            configured = bool(settings.openai_api_key)
            model = settings.openai_model
        elif provider == LLMProvider.GEMINI:
            configured = bool(settings.gemini_api_key)
            model = settings.gemini_model
        elif provider == LLMProvider.GROQ:
            configured = bool(settings.groq_api_key)
            model = settings.groq_model
        else:
            continue
        
        providers.append({
            "name": provider.value,
            "configured": configured,
            "model": model,
            "is_primary": provider == settings.llm_provider
        })
    
    return {
        "providers": providers,
        "fallback_enabled": settings.llm_enable_fallback,
        "fallback_order": settings.llm_fallback_providers.split(",") if settings.llm_fallback_providers else []
    }


@app.get("/templates", tags=["Info"])
async def list_sop_templates():
    """List available SOP templates"""
    templates = [
        {
            "name": SOPTemplate.STANDARD.value,
            "description": "Standard call center compliance checkpoints",
            "checkpoints": ["greeting", "identity_verification", "consent_obtained", "disclosure_provided", "closing"]
        },
        {
            "name": SOPTemplate.BANKING.value,
            "description": "Banking and financial services compliance",
            "checkpoints": ["greeting", "identity_verification", "account_verification", "consent_obtained", "disclosure_provided", "security_reminder", "transaction_confirmation", "closing"]
        },
        {
            "name": SOPTemplate.TELECOM.value,
            "description": "Telecom industry compliance",
            "checkpoints": ["greeting", "identity_verification", "consent_obtained", "plan_explanation", "terms_disclosure", "confirmation", "closing"]
        },
        {
            "name": SOPTemplate.INSURANCE.value,
            "description": "Insurance compliance with coverage details",
            "checkpoints": ["greeting", "identity_verification", "policy_verification", "consent_obtained", "coverage_explanation", "premium_disclosure", "claim_process_explained", "closing"]
        },
        {
            "name": SOPTemplate.HEALTHCARE.value,
            "description": "Healthcare compliance with HIPAA awareness",
            "checkpoints": ["greeting", "identity_verification", "hipaa_consent", "medical_history_review", "confidentiality_reminder", "closing"]
        }
    ]
    
    return {"templates": templates}


# ============= Main Analysis Endpoints =============

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_audio(
    request: AnalyzeRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Submit an audio file for comprehensive analysis.
    
    ### Features
    - Automatic language detection (Hinglish, Tanglish, English)
    - Speaker diarization (Agent vs Customer)
    - Sentiment timeline tracking
    - SOP validation with customizable templates
    - Payment categorization
    - Rejection analysis
    - PII detection and optional redaction
    
    ### Parameters
    - **audio_url**: Public URL of the audio file (MP3, WAV, M4A, OGG)
    - **sop_template**: SOP template to use (standard, banking, telecom, insurance, healthcare)
    - **enable_pii_redaction**: Redact PII in the response
    - **webhook_url**: URL to receive completion notification
    - **metadata**: Custom metadata to include in response
    
    Returns a task_id for polling results.
    """
    logger.info(f"Received analyze request for URL: {request.audio_url[:50]}...")
    
    try:
        # Submit task to Celery with all options
        task = analyze_audio_task.delay(
            audio_url=request.audio_url,
            sop_template=request.sop_template.value if request.sop_template else "standard",
            custom_checkpoints=request.custom_sop_checkpoints,
            enable_pii_redaction=request.enable_pii_redaction or False,
            webhook_url=request.webhook_url,
            metadata=request.metadata
        )
        
        return AnalyzeResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            message="Task submitted successfully. Poll /result/{task_id} for results.",
            estimated_wait_seconds=60
        )
    except Exception as e:
        logger.exception("Failed to submit analysis task")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit task: {str(e)}"
        )


@app.post("/analyze/sync", response_model=TaskResultResponse, tags=["Analysis"])
async def analyze_audio_sync(
    request: AnalyzeRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Synchronous analysis - processes immediately without Celery.
    
    Use this for testing or when Celery/Redis is not available.
    Note: This blocks until processing completes (may take 30-120 seconds).
    """
    logger.info(f"Received sync analyze request for URL: {request.audio_url[:50]}...")
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        from app.utils import download_audio
        from app.stt import get_stt
        from app.processor import get_enhanced_processor
        from app.models import SOPTemplate as SOPTemplateModel
        
        # Step 1: Download audio
        logger.info(f"[{task_id}] Downloading audio...")
        audio_path, download_error = await download_audio(request.audio_url)
        
        if download_error:
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=download_error,
                error_code="DOWNLOAD_FAILED"
            )
        
        # Step 2: Transcribe
        logger.info(f"[{task_id}] Transcribing audio...")
        stt = get_stt()
        transcript, stt_error = stt.transcribe(audio_path)
        
        # Clean up audio file
        import os
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        if stt_error:
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=stt_error,
                error_code="TRANSCRIPTION_FAILED"
            )
        
        # Step 3: Analyze with LLM
        logger.info(f"[{task_id}] Analyzing with LLM...")
        processor = get_enhanced_processor()
        
        sop_template = SOPTemplateModel(request.sop_template.value) if request.sop_template else SOPTemplateModel.STANDARD
        
        analysis, provider_used = processor.analyze_transcript(
            transcript=transcript,
            sop_template=sop_template,
            custom_checkpoints=request.custom_sop_checkpoints
        )
        
        # Step 4: Parse results
        diarization = processor.parse_diarization(analysis)
        sop_validation = processor.parse_sop_validation(analysis, sop_template)
        analytics = processor.parse_analytics(
            analysis, 
            transcript, 
            request.enable_pii_redaction or False
        )
        
        processing_time = time.time() - start_time
        
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            processing_time_seconds=processing_time,
            llm_provider_used=provider_used.value,
            transcript=transcript,
            transcript_redacted=analytics.pii_analysis.redacted_transcript if analytics.pii_analysis else None,
            summary=analysis.get("summary", ""),
            diarized_transcript=diarization.model_dump() if diarization else None,
            sop_validation=sop_validation.model_dump() if sop_validation else None,
            analytics=analytics.model_dump() if analytics else None,
            keywords=analysis.get("keywords_analysis", {}).get("all_keywords", []),
            keywords_analysis=processor.parse_keywords_analysis(analysis).model_dump(),
            metadata=request.metadata,
            overall_confidence=analysis.get("confidence_score", 0.85)
        )
        
    except Exception as e:
        logger.exception(f"[{task_id}] Sync analysis failed")
        processing_time = time.time() - start_time
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            processing_time_seconds=processing_time,
            error=str(e),
            error_code="ANALYSIS_FAILED"
        )


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
async def analyze_batch(
    request: BatchAnalyzeRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Submit multiple audio files for batch analysis.
    
    ### Parameters
    - **audio_urls**: List of audio URLs (max 10)
    - **sop_template**: SOP template to apply to all
    - **enable_pii_redaction**: Redact PII in responses
    - **webhook_url**: URL to receive batch completion notification
    
    Returns batch_id and individual task_ids.
    """
    settings = get_settings()
    
    if len(request.audio_urls) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.max_batch_size}"
        )
    
    logger.info(f"Received batch request for {len(request.audio_urls)} files")
    
    try:
        task = analyze_batch_task.delay(
            audio_urls=request.audio_urls,
            sop_template=request.sop_template.value if request.sop_template else "standard",
            enable_pii_redaction=request.enable_pii_redaction or False,
            webhook_url=request.webhook_url
        )
        
        # Get task IDs from result
        batch_result = task.get(timeout=10)
        
        return BatchAnalyzeResponse(
            batch_id=task.id,
            task_ids=batch_result.get("task_ids", []),
            status=TaskStatus.PENDING,
            message=f"Batch submitted with {len(batch_result.get('task_ids', []))} tasks"
        )
    except Exception as e:
        logger.exception("Failed to submit batch task")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit batch: {str(e)}"
        )


@app.get("/result/{task_id}", response_model=TaskResultResponse, tags=["Analysis"])
async def get_analysis_result(
    task_id: str,
    include_diarization: bool = Query(True, description="Include speaker diarization"),
    include_sentiment_timeline: bool = Query(True, description="Include sentiment timeline"),
    _: bool = Depends(verify_api_key)
):
    """
    Get the result of an analysis task.
    
    Poll this endpoint until status is 'completed' or 'failed'.
    
    ### Parameters
    - **task_id**: The task ID from /analyze
    - **include_diarization**: Include speaker diarization (default: true)
    - **include_sentiment_timeline**: Include sentiment timeline (default: true)
    """
    logger.info(f"Fetching result for task: {task_id}")
    
    # Check stored results
    result = get_result(task_id)
    
    if result:
        # Optionally filter out large fields
        if not include_diarization:
            result.pop("diarized_transcript", None)
            if result.get("analytics"):
                result["analytics"].pop("diarization", None)
        
        if not include_sentiment_timeline:
            if result.get("analytics"):
                result["analytics"].pop("sentiment_timeline", None)
        
        return TaskResultResponse(
            task_id=result.get("task_id", task_id),
            status=TaskStatus(result.get("status", "pending")),
            created_at=datetime.fromisoformat(result["created_at"]) if result.get("created_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(result["completed_at"]) if result.get("completed_at") else None,
            processing_time_seconds=result.get("processing_time_seconds"),
            llm_provider_used=result.get("llm_provider_used"),
            transcript=result.get("transcript"),
            transcript_redacted=result.get("transcript_redacted"),
            summary=result.get("summary"),
            diarized_transcript=result.get("diarized_transcript") if include_diarization else None,
            sop_validation=result.get("sop_validation"),
            analytics=result.get("analytics"),
            keywords=result.get("keywords", []),
            keywords_analysis=result.get("keywords_analysis"),
            metadata=result.get("metadata"),
            error=result.get("error"),
            error_code=result.get("error_code"),
            overall_confidence=result.get("overall_confidence")
        )
    
    # Check Celery task status
    try:
        task = analyze_audio_task.AsyncResult(task_id)
        
        if task.state == "PENDING":
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.PENDING
            )
        elif task.state == "STARTED" or task.state == "PROGRESS":
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.PROCESSING
            )
        elif task.state == "FAILURE":
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(task.info) if task.info else "Unknown error",
                error_code="CELERY_TASK_FAILURE"
            )
        elif task.state == "SUCCESS":
            task_result = task.result or {}
            return TaskResultResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                transcript=task_result.get("transcript"),
                summary=task_result.get("summary"),
                sop_validation=task_result.get("sop_validation"),
                analytics=task_result.get("analytics"),
                keywords=task_result.get("keywords", []),
                error=task_result.get("error")
            )
    except Exception as e:
        logger.warning(f"Error checking Celery task status: {e}")
    
    # Task not found
    raise HTTPException(
        status_code=404,
        detail=f"Task {task_id} not found"
    )


@app.get("/batch/{batch_id}", tags=["Analysis"])
async def get_batch_status(
    batch_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get the status of a batch analysis.
    """
    try:
        from redis import Redis
        settings = get_settings()
        redis_client = Redis.from_url(settings.redis_url)
        
        data = redis_client.get(f"batch:{batch_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        import json
        batch_info = json.loads(data)
        
        # Check status of each task
        completed = 0
        failed = 0
        results = []
        
        for task_id in batch_info.get("task_ids", []):
            result = get_result(task_id)
            if result:
                status = result.get("status")
                if status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
                results.append({
                    "task_id": task_id,
                    "status": status
                })
            else:
                results.append({
                    "task_id": task_id,
                    "status": "pending"
                })
        
        total = len(batch_info.get("task_ids", []))
        
        return {
            "batch_id": batch_id,
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": total - completed - failed,
            "progress_percent": round((completed + failed) / total * 100, 1) if total > 0 else 0,
            "tasks": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching batch status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Error Handlers =============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to ensure clean JSON responses"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "detail": str(exc) if get_settings().debug else "An unexpected error occurred"
        }
    )


# ============= Run with uvicorn =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

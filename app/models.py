"""
Pydantic models for request/response validation
Enhanced with speaker diarization, sentiment timeline, PII detection, and more
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime


# ============= Enums =============

class PaymentType(str, Enum):
    EMI = "emi"
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    DOWN_PAYMENT = "down_payment"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Speaker(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class PIIType(str, Enum):
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    AADHAAR = "aadhaar"
    PAN = "pan"
    BANK_ACCOUNT = "bank_account"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"


class SOPTemplate(str, Enum):
    STANDARD = "standard"
    BANKING = "banking"
    TELECOM = "telecom"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    CUSTOM = "custom"


# ============= Request Models =============

class AnalyzeRequest(BaseModel):
    audio_url: str = Field(..., description="URL of the audio file to analyze")
    sop_template: Optional[SOPTemplate] = Field(default=SOPTemplate.STANDARD, description="SOP template to use")
    custom_sop_checkpoints: Optional[List[str]] = Field(default=None, description="Custom SOP checkpoints")
    enable_pii_redaction: Optional[bool] = Field(default=False, description="Enable PII redaction in transcript")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for completion notification")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata to include in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "https://recordings.exotel.com/exotelrecordings/guvi64/5780094ea05a75c867120809da9a199f.mp3",
                "sop_template": "banking",
                "enable_pii_redaction": False,
                "metadata": {"call_id": "12345", "agent_id": "A001"}
            }
        }


class BatchAnalyzeRequest(BaseModel):
    audio_urls: List[str] = Field(..., min_length=1, max_length=10, description="List of audio URLs to analyze")
    sop_template: Optional[SOPTemplate] = Field(default=SOPTemplate.STANDARD)
    enable_pii_redaction: Optional[bool] = Field(default=False)
    webhook_url: Optional[str] = Field(default=None)


# ============= Speaker Diarization Models =============

class DiarizedSegment(BaseModel):
    speaker: Speaker = Field(..., description="Who is speaking")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL, description="Sentiment of this segment")
    confidence: float = Field(default=0.9, ge=0, le=1, description="Confidence score")


class DiarizedTranscript(BaseModel):
    segments: List[DiarizedSegment] = Field(default_factory=list)
    agent_talk_time_seconds: float = Field(default=0.0)
    customer_talk_time_seconds: float = Field(default=0.0)
    talk_ratio: float = Field(default=1.0, description="Agent to customer talk ratio")
    interruptions_count: int = Field(default=0, description="Number of interruptions detected")


# ============= Sentiment Timeline Models =============

class SentimentPoint(BaseModel):
    timestamp: float = Field(..., description="Time in seconds from call start")
    sentiment: Sentiment
    score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    trigger_phrase: Optional[str] = Field(default=None, description="Phrase that triggered sentiment change")


class SentimentTimeline(BaseModel):
    points: List[SentimentPoint] = Field(default_factory=list)
    overall_sentiment: Sentiment = Field(default=Sentiment.NEUTRAL)
    sentiment_trend: str = Field(default="stable", description="improving, declining, stable, volatile")
    peak_positive_timestamp: Optional[float] = Field(default=None)
    peak_negative_timestamp: Optional[float] = Field(default=None)


# ============= PII Detection Models =============

class PIIEntity(BaseModel):
    type: PIIType
    value: str = Field(..., description="Detected PII value (or redacted)")
    redacted_value: Optional[str] = Field(default=None, description="Redacted version")
    start_position: int = Field(..., description="Start position in transcript")
    end_position: int = Field(..., description="End position in transcript")
    confidence: float = Field(default=0.9, ge=0, le=1)


class PIIAnalysis(BaseModel):
    pii_detected: bool = Field(default=False)
    entities: List[PIIEntity] = Field(default_factory=list)
    redacted_transcript: Optional[str] = Field(default=None, description="Transcript with PII redacted")
    risk_level: str = Field(default="low", description="low, medium, high")


# ============= SOP Validation Models =============

class SOPCheckpoint(BaseModel):
    checkpoint: str = Field(..., description="SOP checkpoint name")
    status: str = Field(..., description="passed/failed/partial/not_applicable")
    details: Optional[str] = Field(default=None, description="Additional context")
    timestamp: Optional[float] = Field(default=None, description="When this checkpoint occurred")
    evidence: Optional[str] = Field(default=None, description="Quote from transcript as evidence")


class SOPValidation(BaseModel):
    overall_compliance: float = Field(..., ge=0, le=100, description="Overall SOP compliance score (0-100)")
    template_used: SOPTemplate = Field(default=SOPTemplate.STANDARD)
    greeting: SOPCheckpoint
    identity_verification: SOPCheckpoint
    consent_obtained: SOPCheckpoint
    disclosure_provided: SOPCheckpoint
    closing: SOPCheckpoint
    # Additional checkpoints for specific templates
    custom_checkpoints: List[SOPCheckpoint] = Field(default_factory=list)
    additional_notes: Optional[str] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


# ============= Payment Analysis Models =============

class PaymentCategory(BaseModel):
    type: PaymentType
    count: int = Field(default=0, ge=0)
    mentions: List[str] = Field(default_factory=list, description="Extracted mentions from transcript")
    total_amount_mentioned: Optional[float] = Field(default=None, description="Total amount if mentioned")
    currency: str = Field(default="INR")


class PaymentAnalysis(BaseModel):
    emi: PaymentCategory = Field(default_factory=lambda: PaymentCategory(type=PaymentType.EMI))
    full_payment: PaymentCategory = Field(default_factory=lambda: PaymentCategory(type=PaymentType.FULL_PAYMENT))
    partial_payment: PaymentCategory = Field(default_factory=lambda: PaymentCategory(type=PaymentType.PARTIAL_PAYMENT))
    down_payment: PaymentCategory = Field(default_factory=lambda: PaymentCategory(type=PaymentType.DOWN_PAYMENT))
    payment_promised: bool = Field(default=False)
    promised_date: Optional[str] = Field(default=None, description="Date customer promised to pay")
    payment_method_mentioned: Optional[str] = Field(default=None, description="UPI, NEFT, Cash, etc.")


# ============= Rejection Analysis Models =============

class RejectionReason(BaseModel):
    category: str = Field(..., description="Category of rejection")
    reason: str = Field(..., description="Specific reason extracted")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    suggested_response: Optional[str] = Field(default=None, description="Suggested agent response")


class RejectionAnalysis(BaseModel):
    has_rejection: bool = Field(default=False)
    reasons: List[RejectionReason] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)
    objection_handling_score: Optional[float] = Field(default=None, ge=0, le=100)


# ============= Call Quality Metrics =============

class AudioQualityMetrics(BaseModel):
    overall_quality: str = Field(default="good", description="good, fair, poor")
    signal_to_noise_ratio: Optional[float] = Field(default=None)
    silence_percentage: float = Field(default=0.0, description="Percentage of call that was silent")
    audio_clipping_detected: bool = Field(default=False)
    background_noise_level: str = Field(default="low", description="low, medium, high")


class CallQualityMetrics(BaseModel):
    audio_quality: AudioQualityMetrics = Field(default_factory=AudioQualityMetrics)
    dead_air_seconds: float = Field(default=0.0, description="Total seconds of silence/dead air")
    hold_time_seconds: float = Field(default=0.0, description="Time customer was on hold")
    average_response_time: float = Field(default=0.0, description="Avg time for agent to respond")
    professionalism_score: Optional[float] = Field(default=None, ge=0, le=100)


# ============= Keywords Models =============

class KeywordCategory(BaseModel):
    category: str = Field(..., description="Category name")
    keywords: List[str] = Field(default_factory=list)
    importance: str = Field(default="medium", description="low, medium, high, critical")


class KeywordsAnalysis(BaseModel):
    all_keywords: List[str] = Field(default_factory=list)
    categorized: List[KeywordCategory] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list, description="Action items extracted")
    follow_up_required: bool = Field(default=False)
    follow_up_reason: Optional[str] = Field(default=None)


# ============= Analytics Models =============

class CallAnalytics(BaseModel):
    call_duration_seconds: Optional[int] = Field(default=None)
    speaker_count: int = Field(default=2)
    language_detected: str = Field(default="hi-en", description="Primary language detected")
    languages_used: List[str] = Field(default_factory=lambda: ["hi-en"])
    overall_sentiment: Sentiment = Field(default=Sentiment.NEUTRAL)
    sentiment_timeline: Optional[SentimentTimeline] = Field(default=None)
    diarization: Optional[DiarizedTranscript] = Field(default=None)
    payment_categorization: PaymentAnalysis = Field(default_factory=PaymentAnalysis)
    rejection_analysis: RejectionAnalysis = Field(default_factory=RejectionAnalysis)
    pii_analysis: Optional[PIIAnalysis] = Field(default=None)
    call_quality: Optional[CallQualityMetrics] = Field(default=None)
    keywords_analysis: Optional[KeywordsAnalysis] = Field(default=None)
    
    # Computed scores
    overall_call_score: Optional[float] = Field(default=None, ge=0, le=100)
    agent_performance_score: Optional[float] = Field(default=None, ge=0, le=100)


# ============= Response Models =============

class AnalyzeResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier for polling")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    message: str = Field(default="Task submitted successfully")
    estimated_wait_seconds: Optional[int] = Field(default=60)


class BatchAnalyzeResponse(BaseModel):
    batch_id: str = Field(..., description="Unique batch identifier")
    task_ids: List[str] = Field(..., description="Individual task IDs")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    message: str = Field(default="Batch submitted successfully")


class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    processing_time_seconds: Optional[float] = Field(default=None)
    
    # LLM provider used
    llm_provider_used: Optional[str] = Field(default=None)
    
    # Core results
    transcript: Optional[str] = Field(default=None)
    transcript_redacted: Optional[str] = Field(default=None, description="Transcript with PII redacted")
    summary: Optional[str] = Field(default=None)
    
    # Diarization
    diarized_transcript: Optional[DiarizedTranscript] = Field(default=None)
    
    # SOP validation
    sop_validation: Optional[SOPValidation] = Field(default=None)
    
    # Analytics
    analytics: Optional[CallAnalytics] = Field(default=None)
    
    # Keywords (enhanced)
    keywords: List[str] = Field(default_factory=list)
    keywords_analysis: Optional[KeywordsAnalysis] = Field(default=None)
    
    # Custom metadata (echoed back)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    
    # Error handling
    error: Optional[str] = Field(default=None)
    error_code: Optional[str] = Field(default=None)
    
    # Confidence
    overall_confidence: Optional[float] = Field(default=None, ge=0, le=1)


class WebhookPayload(BaseModel):
    event: str = Field(default="analysis_complete")
    task_id: str
    status: TaskStatus
    result: Optional[TaskResultResponse] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

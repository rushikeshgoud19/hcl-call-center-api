"""
Enhanced LLM Processor for Call Analysis
Supports: Multi-LLM, Speaker Diarization, Sentiment Timeline, PII Detection, Custom SOP
"""
import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from app.config import get_settings, LLMProvider
from app.llm_providers import get_multi_llm_analyzer
from app.models import (
    SOPValidation, SOPCheckpoint, CallAnalytics, PaymentAnalysis,
    PaymentCategory, PaymentType, RejectionAnalysis, RejectionReason,
    DiarizedTranscript, DiarizedSegment, Speaker, Sentiment,
    SentimentTimeline, SentimentPoint, PIIAnalysis, PIIEntity, PIIType,
    KeywordsAnalysis, KeywordCategory, CallQualityMetrics, AudioQualityMetrics,
    SOPTemplate
)

logger = logging.getLogger(__name__)

# ============= SYSTEM PROMPTS =============

SYSTEM_PROMPT = """You are an expert Call Quality Auditor for Indian financial services call centers.
You analyze call transcripts in Hindi (Hinglish), Tamil (Tanglish), Telugu, and English.
You understand code-switching between languages and can extract meaning accurately from mixed-language conversations.
Your analysis must be objective, accurate, and based solely on the transcript content.
Always respond with valid JSON matching the exact schema requested.
Never make up information not present in the transcript."""


# ============= ANALYSIS PROMPTS =============

COMPREHENSIVE_ANALYSIS_PROMPT = """Analyze the following call center transcript comprehensively.

TRANSCRIPT:
{transcript}

Provide a COMPLETE analysis in the following JSON format:

{{
    "summary": "A 2-3 sentence summary of the call in English",
    
    "diarization": {{
        "segments": [
            {{
                "speaker": "agent|customer",
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "exact text spoken",
                "sentiment": "positive|neutral|negative",
                "confidence": 0.9
            }}
        ],
        "agent_talk_time_seconds": 120.0,
        "customer_talk_time_seconds": 80.0,
        "talk_ratio": 1.5,
        "interruptions_count": 2
    }},
    
    "sentiment_timeline": {{
        "points": [
            {{
                "timestamp": 0.0,
                "sentiment": "neutral",
                "score": 0.0,
                "trigger_phrase": "phrase that changed sentiment"
            }}
        ],
        "overall_sentiment": "positive|neutral|negative",
        "sentiment_trend": "improving|declining|stable|volatile",
        "peak_positive_timestamp": null,
        "peak_negative_timestamp": null
    }},
    
    "sop_validation": {{
        "overall_compliance": 85.0,
        "template_used": "{sop_template}",
        "greeting": {{
            "checkpoint": "greeting",
            "status": "passed|failed|partial|not_applicable",
            "details": "explanation",
            "timestamp": 0.0,
            "evidence": "exact quote from transcript"
        }},
        "identity_verification": {{
            "checkpoint": "identity_verification",
            "status": "passed|failed|partial|not_applicable",
            "details": "Did agent verify customer identity?",
            "timestamp": null,
            "evidence": null
        }},
        "consent_obtained": {{
            "checkpoint": "consent_obtained",
            "status": "passed|failed|partial|not_applicable",
            "details": "Was consent taken?",
            "timestamp": null,
            "evidence": null
        }},
        "disclosure_provided": {{
            "checkpoint": "disclosure_provided",
            "status": "passed|failed|partial|not_applicable",
            "details": "Were disclosures made?",
            "timestamp": null,
            "evidence": null
        }},
        "closing": {{
            "checkpoint": "closing",
            "status": "passed|failed|partial|not_applicable",
            "details": "Was call closed properly?",
            "timestamp": null,
            "evidence": null
        }},
        "custom_checkpoints": [],
        "additional_notes": "other SOP observations",
        "recommendations": ["improvement suggestion 1", "improvement suggestion 2"]
    }},
    
    "payment_categorization": {{
        "emi": {{
            "type": "emi",
            "count": 0,
            "mentions": [],
            "total_amount_mentioned": null,
            "currency": "INR"
        }},
        "full_payment": {{
            "type": "full_payment",
            "count": 0,
            "mentions": [],
            "total_amount_mentioned": null,
            "currency": "INR"
        }},
        "partial_payment": {{
            "type": "partial_payment",
            "count": 0,
            "mentions": [],
            "total_amount_mentioned": null,
            "currency": "INR"
        }},
        "down_payment": {{
            "type": "down_payment",
            "count": 0,
            "mentions": [],
            "total_amount_mentioned": null,
            "currency": "INR"
        }},
        "payment_promised": false,
        "promised_date": null,
        "payment_method_mentioned": null
    }},
    
    "rejection_analysis": {{
        "has_rejection": false,
        "reasons": [
            {{
                "category": "financial_constraint|timing_issue|product_issue|personal_reason|dispute|network_issue|callback_requested|other",
                "reason": "specific reason",
                "confidence": 0.9,
                "suggested_response": "how agent could handle this"
            }}
        ],
        "summary": "rejection summary",
        "objection_handling_score": 75.0
    }},
    
    "pii_detected": {{
        "pii_detected": true|false,
        "entities": [
            {{
                "type": "phone_number|email|aadhaar|pan|bank_account|credit_card|address|name|date_of_birth",
                "value": "detected value",
                "start_position": 0,
                "end_position": 10,
                "confidence": 0.95
            }}
        ],
        "risk_level": "low|medium|high"
    }},
    
    "keywords_analysis": {{
        "all_keywords": ["keyword1", "keyword2"],
        "categorized": [
            {{
                "category": "payment_related|complaint|positive_feedback|urgency|escalation|technical",
                "keywords": ["keyword1"],
                "importance": "low|medium|high|critical"
            }}
        ],
        "action_items": ["action item extracted from call"],
        "follow_up_required": true|false,
        "follow_up_reason": "reason for follow up"
    }},
    
    "call_metrics": {{
        "language_detected": "hi-en|ta-en|te-en|en|hi|ta",
        "languages_used": ["hi-en", "en"],
        "overall_call_score": 78.0,
        "agent_performance_score": 82.0,
        "professionalism_score": 85.0
    }},
    
    "confidence_score": 0.92
}}

RULES:
1. Extract ACTUAL phrases/quotes from transcript for evidence fields
2. Identify speakers (agent vs customer) based on context and language patterns
3. Track sentiment changes throughout the call with timestamps
4. Detect ALL PII (phone numbers, Aadhaar format XXXX-XXXX-XXXX, PAN format ABCDE1234F, etc.)
5. Consider Hinglish/Tanglish: "paisa", "EMI", "kist", "thavanai", "payment", etc.
6. For SOP, only mark "passed" with clear evidence
7. Extract meaningful keywords: payment_pending, emi_overdue, customer_angry, callback_requested
8. Identify action items and follow-up requirements

Respond ONLY with valid JSON."""


# ============= SOP TEMPLATES =============

SOP_TEMPLATES = {
    SOPTemplate.STANDARD: [
        "greeting", "identity_verification", "consent_obtained", 
        "disclosure_provided", "closing"
    ],
    SOPTemplate.BANKING: [
        "greeting", "identity_verification", "account_verification",
        "consent_obtained", "disclosure_provided", "security_reminder",
        "transaction_confirmation", "closing"
    ],
    SOPTemplate.TELECOM: [
        "greeting", "identity_verification", "consent_obtained",
        "plan_explanation", "terms_disclosure", "confirmation", "closing"
    ],
    SOPTemplate.INSURANCE: [
        "greeting", "identity_verification", "policy_verification",
        "consent_obtained", "coverage_explanation", "premium_disclosure",
        "claim_process_explained", "closing"
    ],
    SOPTemplate.HEALTHCARE: [
        "greeting", "identity_verification", "hipaa_consent",
        "medical_history_review", "confidentiality_reminder", "closing"
    ]
}


class EnhancedProcessor:
    """
    Enhanced processor with multi-LLM support and comprehensive analysis
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.multi_llm = get_multi_llm_analyzer()
    
    def analyze_transcript(
        self, 
        transcript: str,
        sop_template: SOPTemplate = SOPTemplate.STANDARD,
        custom_checkpoints: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], LLMProvider]:
        """
        Comprehensive transcript analysis using multi-LLM with fallback.
        Returns: (analysis_dict, provider_used)
        """
        prompt = COMPREHENSIVE_ANALYSIS_PROMPT.replace("{sop_template}", sop_template.value)
        
        analysis, provider = self.multi_llm.analyze_with_fallback(
            transcript=transcript,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT
        )
        
        return analysis, provider
    
    def parse_diarization(self, analysis: Dict[str, Any]) -> Optional[DiarizedTranscript]:
        """Parse speaker diarization from analysis"""
        diar_data = analysis.get("diarization")
        if not diar_data:
            return None
        
        segments = []
        for seg in diar_data.get("segments", []):
            try:
                segments.append(DiarizedSegment(
                    speaker=Speaker(seg.get("speaker", "unknown")),
                    start_time=seg.get("start_time", 0.0),
                    end_time=seg.get("end_time", 0.0),
                    text=seg.get("text", ""),
                    sentiment=Sentiment(seg.get("sentiment", "neutral")),
                    confidence=seg.get("confidence", 0.9)
                ))
            except Exception as e:
                logger.warning(f"Failed to parse segment: {e}")
                continue
        
        return DiarizedTranscript(
            segments=segments,
            agent_talk_time_seconds=diar_data.get("agent_talk_time_seconds", 0.0),
            customer_talk_time_seconds=diar_data.get("customer_talk_time_seconds", 0.0),
            talk_ratio=diar_data.get("talk_ratio", 1.0),
            interruptions_count=diar_data.get("interruptions_count", 0)
        )
    
    def parse_sentiment_timeline(self, analysis: Dict[str, Any]) -> Optional[SentimentTimeline]:
        """Parse sentiment timeline from analysis"""
        sent_data = analysis.get("sentiment_timeline")
        if not sent_data:
            return None
        
        points = []
        for pt in sent_data.get("points", []):
            try:
                points.append(SentimentPoint(
                    timestamp=pt.get("timestamp", 0.0),
                    sentiment=Sentiment(pt.get("sentiment", "neutral")),
                    score=pt.get("score", 0.0),
                    trigger_phrase=pt.get("trigger_phrase")
                ))
            except Exception as e:
                logger.warning(f"Failed to parse sentiment point: {e}")
                continue
        
        return SentimentTimeline(
            points=points,
            overall_sentiment=Sentiment(sent_data.get("overall_sentiment", "neutral")),
            sentiment_trend=sent_data.get("sentiment_trend", "stable"),
            peak_positive_timestamp=sent_data.get("peak_positive_timestamp"),
            peak_negative_timestamp=sent_data.get("peak_negative_timestamp")
        )
    
    def parse_pii_analysis(self, analysis: Dict[str, Any], transcript: str, redact: bool = False) -> PIIAnalysis:
        """Parse PII detection and optionally redact"""
        pii_data = analysis.get("pii_detected", {})
        
        entities = []
        for ent in pii_data.get("entities", []):
            try:
                pii_type = PIIType(ent.get("type", "name"))
                value = ent.get("value", "")
                
                # Generate redacted value
                redacted = self._redact_value(pii_type, value)
                
                entities.append(PIIEntity(
                    type=pii_type,
                    value=value if not redact else redacted,
                    redacted_value=redacted,
                    start_position=ent.get("start_position", 0),
                    end_position=ent.get("end_position", 0),
                    confidence=ent.get("confidence", 0.9)
                ))
            except Exception as e:
                logger.warning(f"Failed to parse PII entity: {e}")
                continue
        
        # Generate redacted transcript if requested
        redacted_transcript = None
        if redact and entities:
            redacted_transcript = self._redact_transcript(transcript, entities)
        
        return PIIAnalysis(
            pii_detected=pii_data.get("pii_detected", len(entities) > 0),
            entities=entities,
            redacted_transcript=redacted_transcript,
            risk_level=pii_data.get("risk_level", "low" if not entities else "medium")
        )
    
    def _redact_value(self, pii_type: PIIType, value: str) -> str:
        """Redact a PII value based on its type"""
        if pii_type == PIIType.PHONE_NUMBER:
            return "XXX-XXX-" + value[-4:] if len(value) >= 4 else "XXXX"
        elif pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                return parts[0][:2] + "***@" + parts[1]
            return "***@***.com"
        elif pii_type == PIIType.AADHAAR:
            return "XXXX-XXXX-" + value[-4:] if len(value) >= 4 else "XXXX-XXXX-XXXX"
        elif pii_type == PIIType.PAN:
            return "XXXXX" + value[-5:] if len(value) >= 5 else "XXXXXXXXXX"
        elif pii_type == PIIType.CREDIT_CARD:
            return "XXXX-XXXX-XXXX-" + value[-4:] if len(value) >= 4 else "XXXX"
        elif pii_type == PIIType.BANK_ACCOUNT:
            return "XXXXX" + value[-4:] if len(value) >= 4 else "XXXXX"
        else:
            return "[REDACTED]"
    
    def _redact_transcript(self, transcript: str, entities: List[PIIEntity]) -> str:
        """Redact all PII from transcript"""
        redacted = transcript
        # Sort by position descending to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x.start_position, reverse=True)
        
        for entity in sorted_entities:
            if entity.redacted_value:
                redacted = (
                    redacted[:entity.start_position] + 
                    entity.redacted_value + 
                    redacted[entity.end_position:]
                )
        
        return redacted
    
    def parse_keywords_analysis(self, analysis: Dict[str, Any]) -> KeywordsAnalysis:
        """Parse keywords analysis"""
        kw_data = analysis.get("keywords_analysis", {})
        
        categorized = []
        for cat in kw_data.get("categorized", []):
            categorized.append(KeywordCategory(
                category=cat.get("category", "general"),
                keywords=cat.get("keywords", []),
                importance=cat.get("importance", "medium")
            ))
        
        return KeywordsAnalysis(
            all_keywords=kw_data.get("all_keywords", []),
            categorized=categorized,
            action_items=kw_data.get("action_items", []),
            follow_up_required=kw_data.get("follow_up_required", False),
            follow_up_reason=kw_data.get("follow_up_reason")
        )
    
    def parse_sop_validation(
        self, 
        analysis: Dict[str, Any],
        template: SOPTemplate = SOPTemplate.STANDARD
    ) -> SOPValidation:
        """Parse SOP validation from analysis dict"""
        sop_data = analysis.get("sop_validation", {})
        
        def parse_checkpoint(data: Dict, name: str) -> SOPCheckpoint:
            return SOPCheckpoint(
                checkpoint=name,
                status=data.get("status", "failed"),
                details=data.get("details"),
                timestamp=data.get("timestamp"),
                evidence=data.get("evidence")
            )
        
        # Parse custom checkpoints
        custom_checkpoints = []
        for cp in sop_data.get("custom_checkpoints", []):
            custom_checkpoints.append(parse_checkpoint(cp, cp.get("checkpoint", "custom")))
        
        return SOPValidation(
            overall_compliance=sop_data.get("overall_compliance", 0),
            template_used=template,
            greeting=parse_checkpoint(sop_data.get("greeting", {}), "greeting"),
            identity_verification=parse_checkpoint(sop_data.get("identity_verification", {}), "identity_verification"),
            consent_obtained=parse_checkpoint(sop_data.get("consent_obtained", {}), "consent_obtained"),
            disclosure_provided=parse_checkpoint(sop_data.get("disclosure_provided", {}), "disclosure_provided"),
            closing=parse_checkpoint(sop_data.get("closing", {}), "closing"),
            custom_checkpoints=custom_checkpoints,
            additional_notes=sop_data.get("additional_notes"),
            recommendations=sop_data.get("recommendations", [])
        )
    
    def parse_payment_analysis(self, analysis: Dict[str, Any]) -> PaymentAnalysis:
        """Parse payment analysis"""
        pay_data = analysis.get("payment_categorization", {})
        
        def parse_category(data: Dict, ptype: PaymentType) -> PaymentCategory:
            return PaymentCategory(
                type=ptype,
                count=data.get("count", 0),
                mentions=data.get("mentions", []),
                total_amount_mentioned=data.get("total_amount_mentioned"),
                currency=data.get("currency", "INR")
            )
        
        return PaymentAnalysis(
            emi=parse_category(pay_data.get("emi", {}), PaymentType.EMI),
            full_payment=parse_category(pay_data.get("full_payment", {}), PaymentType.FULL_PAYMENT),
            partial_payment=parse_category(pay_data.get("partial_payment", {}), PaymentType.PARTIAL_PAYMENT),
            down_payment=parse_category(pay_data.get("down_payment", {}), PaymentType.DOWN_PAYMENT),
            payment_promised=pay_data.get("payment_promised", False),
            promised_date=pay_data.get("promised_date"),
            payment_method_mentioned=pay_data.get("payment_method_mentioned")
        )
    
    def parse_rejection_analysis(self, analysis: Dict[str, Any]) -> RejectionAnalysis:
        """Parse rejection analysis"""
        rej_data = analysis.get("rejection_analysis", {})
        
        reasons = []
        for r in rej_data.get("reasons", []):
            reasons.append(RejectionReason(
                category=r.get("category", "other"),
                reason=r.get("reason", ""),
                confidence=r.get("confidence", 0.5),
                suggested_response=r.get("suggested_response")
            ))
        
        return RejectionAnalysis(
            has_rejection=rej_data.get("has_rejection", False),
            reasons=reasons,
            summary=rej_data.get("summary"),
            objection_handling_score=rej_data.get("objection_handling_score")
        )
    
    def parse_analytics(
        self, 
        analysis: Dict[str, Any],
        transcript: str,
        enable_pii_redaction: bool = False
    ) -> CallAnalytics:
        """Parse complete call analytics"""
        metrics = analysis.get("call_metrics", {})
        
        return CallAnalytics(
            speaker_count=2,
            language_detected=metrics.get("language_detected", "hi-en"),
            languages_used=metrics.get("languages_used", ["hi-en"]),
            overall_sentiment=Sentiment(
                analysis.get("sentiment_timeline", {}).get("overall_sentiment", "neutral")
            ),
            sentiment_timeline=self.parse_sentiment_timeline(analysis),
            diarization=self.parse_diarization(analysis),
            payment_categorization=self.parse_payment_analysis(analysis),
            rejection_analysis=self.parse_rejection_analysis(analysis),
            pii_analysis=self.parse_pii_analysis(analysis, transcript, enable_pii_redaction),
            keywords_analysis=self.parse_keywords_analysis(analysis),
            overall_call_score=metrics.get("overall_call_score"),
            agent_performance_score=metrics.get("agent_performance_score")
        )


# Singleton instance
_processor: Optional[EnhancedProcessor] = None


def get_enhanced_processor() -> EnhancedProcessor:
    """Get or create EnhancedProcessor instance"""
    global _processor
    if _processor is None:
        _processor = EnhancedProcessor()
    return _processor

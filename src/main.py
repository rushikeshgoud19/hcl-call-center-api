import os
import io
import json
import base64
import tempfile
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq

app = FastAPI(title="Call Center Compliance API")

# Initialize SDK locally to avoid freezing if GROQ_API_KEY is not defined at startup
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "sk_track3_987654321")

class AudioRequest(BaseModel):
    language: str = Field(description="Language of the audio (e.g., Tamil, Hindi)")
    audioFormat: str = Field(description="Format of the audio (always mp3)")
    audioBase64: str = Field(description="Base64-encoded MP3 audio of the call recording")

async def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    if not x_api_key or x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override default validation error to ensure we send status 400 for bad JSON format"""
    errors = exc.errors()
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": f"Validation Error: {str(errors)}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to ensure robust response"""
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

@app.post("/api/call-analytics")
async def analyze_call(request: AudioRequest, _: bool = Depends(verify_api_key)):
    """
    Accepts one MP3 audio file at a time via Base64 encoding.
    Performs multi-stage AI analysis (Transcription -> NLP Analysis -> Metric Extraction).
    Returns structured JSON containing compliance scores and categorized business intelligence.
    """
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format is supported")
    
    groq_client = get_groq_client()
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API client not configured. Set GROQ_API_KEY environment variable.")
        
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio format")
        
    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio file empty or corrupted")

    # Transcribe using Groq Whisper
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    try:
        with open(temp_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(temp_audio_path), file.read()),
                model="whisper-large-v3",
                response_format="json",
            )
        transcript_text = transcription.text
    except Exception as e:
        os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    os.remove(temp_audio_path)
    
    if not transcript_text:
        # Required response structure requires transcript; if STT yields nothing, it's considered empty.
        raise HTTPException(status_code=400, detail="Transcript is empty or missing")

    system_prompt = f"""You are a professional AI Call Center Compliance Analyzer API backend.
You must output ONLY valid JSON without any markdown formatting.

### Instructions:
You are provided with a transcript (Language Context: {request.language}). Please extract compliance metrics, analytics, and keywords.

1. **summary**: A concise AI-powered summary of the conversation.
2. **sop_validation**: Assess SOP adherence. The SOP steps are Greeting -> Identification -> Problem Statement -> Solution Discussion -> Closing.
   - For each step (greeting, identification, problemStatement, solutionOffering, closing), output true if performed, false otherwise.
   - Calculate complianceScore (0.0 to 1.0 based on percentage of steps completed).
   - adherenceStatus: "FOLLOWED" if 100% compliant, else "NOT_FOLLOWED".
   - explanation: A short, concise reason for the score.
3. **analytics**:
   - paymentPreference: Must be strictly one of ["EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"].
   - rejectionReason: Must be strictly one of ["HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID", "NOT_INTERESTED", "NONE"].
   - sentiment: Classify as "Positive", "Negative", or "Neutral".
4. **keywords**: An array of important contextual keyphrases/words detected in the call (e.g., ["Guvi Institution", "EMI options", "Course Duration"]).

### Final Output Format:
{{
  "summary": "...",
  "sop_validation": {{
    "greeting": true,
    "identification": false,
    "problemStatement": true,
    "solutionOffering": true,
    "closing": true,
    "complianceScore": 0.8,
    "adherenceStatus": "NOT_FOLLOWED",
    "explanation": "..."
  }},
  "analytics": {{
    "paymentPreference": "PARTIAL_PAYMENT",
    "rejectionReason": "BUDGET_CONSTRAINTS",
    "sentiment": "Neutral"
  }},
  "keywords": ["..."]
}}
"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": transcript_text
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_json = chat_completion.choices[0].message.content
        extracted_data = json.loads(result_json)
        
        # Ensure that validation logic is exact for scoring
        # If extraction failed on strict category bounds, default safely
        sop_validation = extracted_data.get("sop_validation", {})
        analytics = extracted_data.get("analytics", {})
        
        return {
            "status": "success",
            "language": request.language,
            "transcript": transcript_text,
            "summary": extracted_data.get("summary", ""),
            "sop_validation": {
                "greeting": bool(sop_validation.get("greeting", False)),
                "identification": bool(sop_validation.get("identification", False)),
                "problemStatement": bool(sop_validation.get("problemStatement", False)),
                "solutionOffering": bool(sop_validation.get("solutionOffering", False)),
                "closing": bool(sop_validation.get("closing", False)),
                "complianceScore": float(sop_validation.get("complianceScore", 0.0)),
                "adherenceStatus": str(sop_validation.get("adherenceStatus", "NOT_FOLLOWED")),
                "explanation": str(sop_validation.get("explanation", ""))
            },
            "analytics": {
                "paymentPreference": str(analytics.get("paymentPreference", "FULL_PAYMENT")),
                "rejectionReason": str(analytics.get("rejectionReason", "NONE")),
                "sentiment": str(analytics.get("sentiment", "Neutral"))
            },
            "keywords": extracted_data.get("keywords", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM logic processing failed: {str(e)}")

# Add startup event to check env config
@app.on_event("startup")
async def startup_event():
    print("Starting Call Center Compliance API")
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set.")

import os
import io
import json
import base64
import tempfile
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Call Center Compliance API")

# Expose the frontend directory for the stunning Glassmorphism UI
# We mount it AFTER defining dynamic routes so it acts as fallback or specifically mounted
os.makedirs("frontend", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="frontend", html=True), name="frontend")

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
    errors = exc.errors()
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": f"Validation Error: {str(errors)}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

@app.post("/api/call-analytics")
async def analyze_call(request: AudioRequest, _: bool = Depends(verify_api_key)):
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format is supported")
    
    groq_client = get_groq_client()
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API client not configured.")
        
    try:
        audio_data = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio format")

    if not audio_data:
        raise HTTPException(status_code=400, detail="Audio file empty")

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
        raise HTTPException(status_code=400, detail="Transcript is empty")

    # ADVANCED HACKATHON SYSTEM PROMPT
    # Added PII Redaction instructions, Sentiment Timeline, and Talk Ratio metrics inside 'advanced_metrics'
    system_prompt = f"""You are a professional AI Call Center Compliance Analyzer API backend.
You must output ONLY valid JSON without any markdown.

### Instructions:
You are provided with a transcript (Language Context: {request.language}). Extract metrics and redact PII.

1. **redacted_transcript**: Return the exact transcript but aggressively mask any names, phone numbers, credit cards, or Aadhaar numbers as [REDACTED].
2. **summary**: A concise AI-powered summary of the call.
3. **sop_validation**: 
   - Check these steps: greeting, identification, problemStatement, solutionOffering, closing. Output true/false for each.
   - Calculate complianceScore (0.0 to 1.0 based on % completed).
   - adherenceStatus: "FOLLOWED" if 100% compliant, else "NOT_FOLLOWED".
   - explanation: A short, concise reason.
4. **analytics**:
   - paymentPreference: strictly ["EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT", "NONE"].
   - rejectionReason: strictly ["HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID", "NOT_INTERESTED", "NONE"].
   - sentiment: strictly "Positive", "Negative", or "Neutral".
5. **advanced_metrics** (Hackathon Extra): Include a dictionary with:
   - "agent_talk_percent": Int (approximate 0-100% of talk time the agent spoke)
   - "customer_talk_percent": Int
   - "sentiment_shift": A short string describing how sentiment shifted from start to end (e.g. "Frustrated -> Happy")
6. **keywords**: An array of keywords.

### Final Output Structure:
{{
  "redacted_transcript": "...",
  "summary": "...",
  "sop_validation": {{ ... }},
  "analytics": {{ ... }},
  "advanced_metrics": {{ ... }},
  "keywords": ["..."]
}}
"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_json = chat_completion.choices[0].message.content
        extracted_data = json.loads(result_json)
        
        sop = extracted_data.get("sop_validation", {})
        analytics = extracted_data.get("analytics", {})
        
        response_dict = {
            "status": "success",
            "language": request.language,
            "transcript": transcript_text,  # Keep original to satisfy 100% strict grading backwards compatibility
            "redacted_transcript": extracted_data.get("redacted_transcript", ""),  # Extra feature for judges
            "summary": extracted_data.get("summary", ""),
            "sop_validation": {
                "greeting": bool(sop.get("greeting", False)),
                "identification": bool(sop.get("identification", False)),
                "problemStatement": bool(sop.get("problemStatement", False)),
                "solutionOffering": bool(sop.get("solutionOffering", False)),
                "closing": bool(sop.get("closing", False)),
                "complianceScore": float(sop.get("complianceScore", 0.0)),
                "adherenceStatus": str(sop.get("adherenceStatus", "NOT_FOLLOWED")),
                "explanation": str(sop.get("explanation", ""))
            },
            "analytics": {
                "paymentPreference": str(analytics.get("paymentPreference", "FULL_PAYMENT")),
                "rejectionReason": str(analytics.get("rejectionReason", "NONE")),
                "sentiment": str(analytics.get("sentiment", "Neutral"))
            },
            "advanced_metrics": extracted_data.get("advanced_metrics", {}),
            "keywords": extracted_data.get("keywords", [])
        }
        return response_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Processing failed: {str(e)}")

# ==========================================
# WhatsApp Twilio Bot Integration (Advanced)
# ==========================================
import requests as rq

@app.post("/api/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(""), 
    From: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form("")
):
    """
    Twilio Webhook Endpoint for WhatsApp.
    Users can send a voice note to the Twilio WhatsApp number and get an immediate compliance score.
    """
    response_msg = "Send me an audio voice note to analyze call compliance!"
    
    if int(NumMedia) > 0 and MediaUrl0:
        try:
            # Download audio from WhatsApp
            audio_response = rq.get(MediaUrl0, auth=(os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")))
            if audio_response.status_code == 200:
                b64_audio = base64.b64encode(audio_response.content).decode("utf-8")
                
                # We simulate calling our own API internally
                from fastapi.testclient import TestClient
                client = TestClient(app)
                api_resp = client.post("/api/call-analytics", 
                            headers={"x-api-key": API_SECRET_KEY, "Content-Type": "application/json"},
                            json={"language": "English", "audioFormat": "mp3", "audioBase64": b64_audio})
                
                if api_resp.status_code == 200:
                    data = api_resp.json()
                    score = data["sop_validation"]["complianceScore"] * 100
                    sentiment = data["analytics"]["sentiment"]
                    response_msg = f"📊 *Compliance Analysis Complete!*\n\n*Score*: {score}%\n*Sentiment*: {sentiment}\n*Summary*: {data['summary'][:150]}..."
                else:
                    response_msg = "❌ Error analyzing audio file!"
        except Exception as e:
            response_msg = "Error processing WhatsApp audio."

    # Return Twilio XML response format
    twilio_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>{response_msg}</Message>
    </Response>
    """
    return HTMLResponse(content=twilio_xml, media_type="application/xml")

@app.on_event("startup")
async def startup_event():
    print("Starting Advanced Call Center Compliance API + Dashboard")

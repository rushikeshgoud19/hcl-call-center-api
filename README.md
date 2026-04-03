# Call Center Compliance API

## Description
This project implements the Call Center Compliance API, designed to accept call recordings (via Base64 MP3 structure) and perform an AI-driven multi-stage transcription and NLP analysis pipeline. 

The strategy behind the solution focuses on **extreme performance and strict response validation**. By leveraging the **Groq API** (which powers LPU inference engines instead of GPUs), the application ensures incredibly fast STT via `whisper-large-v3` and deep semantic structuring via `llama-3.3-70b-versatile` all within a single request. 

The API guarantees the strict JSON response format demanded by the classification rules, extracting metrics like SOP validation scores, semantic summaries, and categorizations like `paymentPreference` and `rejectionReason`.

## Tech Stack
- **Language/Framework**: Python 3.9+ / FastAPI
- **Key libraries**: `fastapi`, `uvicorn`, `pydantic`, `groq`
- **LLM/AI models used**: 
  - **Speech-to-Text**: `whisper-large-v3` (via Groq)
  - **NLP & Classification**: `llama-3.3-70b-versatile` (via Groq)

## Setup Instructions
1. Clone the repository
   ```bash
   git clone <your-repo-link>
   cd <your-repo-directory>
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables
   ```bash
   cp .env.example .env
   # Open .env and add your GROQ_API_KEY and API_SECRET_KEY
   ```
4. Run the application
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Usage

Example cURL Request:
```bash
curl -X POST http://localhost:8000/api/call-analytics \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_track3_987654321" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "<YOUR_BASE64_STRING>"
  }'
```

## Approach
Our implementation uses a two-step AI pipeline on Groq:
1. **Audio Decoding and STT**: The Base64 string is decoded, temporarily saved, and sent to Groq's high-speed Whisper Large v3 model which excels in code-mixed languages like Tanglish and Hinglish.
2. **Schema-Enforced Analysis**: The parsed string is sent as context into Llama-3.3-70b-versatile. A carefully constructed system prompt combined with native `json_object` enforcement guarantees that the Large Language Model generates exclusively a conformant payload. True/False booleans for SOP constraints are extracted deterministically.

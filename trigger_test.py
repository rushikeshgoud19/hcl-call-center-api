import json
import base64
import requests
from gtts import gTTS

print("1. Generating synthetic MP3 test file...")
tts = gTTS("Hello, is this Manikandan? Yes. Great, I am calling about your EMI payment structure. I can pay the first part now, but not the full amount. Okay, partial payment is fine. Thank you, goodbye.", lang='en')
tts.save("synthetic_call.mp3")

print("2. Encoding MP3 to Base64...")
with open("synthetic_call.mp3", "rb") as audio_file:
    encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')

payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": encoded_string
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": "sk_track3_987654321"
}

# print("\n--- Testing Local API (localhost:8000) ---")
# try:
#     res_local = requests.post("http://localhost:8000/api/call-analytics", json=payload, headers=headers)
#     print(f"Status Code: {res_local.status_code}")
#     print(json.dumps(res_local.json(), indent=2))
# except Exception as e:
#     print(f"Failed to connect to local API: {e}")

print("\n--- Testing Deployed API (Render) ---")
try:
    res_render = requests.post("https://hcl-call-center-api.onrender.com/api/call-analytics", json=payload, headers=headers)
    print(f"Status Code: {res_render.status_code}")
    print(json.dumps(res_render.json(), indent=2))
except Exception as e:
    print(f"Failed to connect to Render API: {e}")

import requests
import base64

url = "http://localhost:8000/api/v1/predict"
# Minimal valid WAV header + empty data
dummy_wav = base64.b64encode(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00').decode('utf-8')

payload = {
    "audio_base64": dummy_wav,
    "sample_rate": 16000
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

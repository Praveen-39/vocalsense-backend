import requests

url = "http://localhost:8000/api/v1/predict"
# Invalid base64
payload = {
    "audio_base64": "NotABase64String!!!",
    "sample_rate": 16000
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

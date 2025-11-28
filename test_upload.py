"""
Test script to verify file upload endpoint works
"""
import requests
import sys

def test_file_upload(file_path):
    """Test uploading an audio file to the API"""
    
    url = "http://localhost:8000/api/v1/predict-file"
    
    try:
        # Open and send file
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'audio/wav')}
            
            print(f"Uploading file: {file_path}")
            response = requests.post(url, files=files)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✓ Success!")
                print(f"Transcript: {data['transcript']}")
                print(f"Emotion: {data['emotion']} ({data['emotion_confidence']})")
                print(f"Sarcasm: {data['sarcasm']} ({data['sarcasm_score']})")
            else:
                print(f"\n✗ Error: {response.text}")
                
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file_upload(sys.argv[1])
    else:
        print("Usage: python test_upload.py <audio_file_path>")
        print("Example: python test_upload.py sample.wav")

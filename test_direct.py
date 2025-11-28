"""
Direct test of the inference function
"""
from app.real_inference import run_inference

# Read test audio file
with open('test_audio.wav', 'rb') as f:
    audio_bytes = f.read()

print(f"Testing with {len(audio_bytes)} bytes of audio data\n")

# Run inference
emotion, em_conf, sarcasm, sarcasm_score, transcript = run_inference(audio_bytes, sample_rate=None)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"Transcript: {transcript}")
print(f"Emotion: {emotion} ({em_conf})")
print(f"Sarcasm: {sarcasm} ({sarcasm_score})")
print(f"{'='*60}")

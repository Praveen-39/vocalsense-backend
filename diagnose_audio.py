"""
Diagnostic script to test audio file loading
"""
import sys
import io
import librosa
import soundfile as sf

def diagnose_audio_file(file_path):
    """Diagnose issues with audio file"""
    
    print(f"Diagnosing: {file_path}\n")
    
    # Test 1: Read file bytes
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        print(f"✓ File read successfully: {len(audio_bytes)} bytes")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return
    
    # Test 2: Load with librosa (auto sample rate)
    try:
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        print(f"✓ Librosa load (auto SR): {len(audio_data)} samples, {sr}Hz, {len(audio_data)/sr:.2f}s")
    except Exception as e:
        print(f"✗ Librosa load failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Load with soundfile
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        print(f"✓ Soundfile read: {len(data)} samples, {sr}Hz")
    except Exception as e:
        print(f"✗ Soundfile read failed: {e}")
    
    # Test 4: File info
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        print(f"✓ File info:")
        print(f"  - Format: {info.format}")
        print(f"  - Subtype: {info.subtype}")
        print(f"  - Channels: {info.channels}")
        print(f"  - Sample rate: {info.samplerate}Hz")
        print(f"  - Duration: {info.duration:.2f}s")
    except Exception as e:
        print(f"✗ File info failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose_audio_file(sys.argv[1])
    else:
        print("Usage: python diagnose_audio.py <audio_file>")
        print("Example: python diagnose_audio.py sample.wav")

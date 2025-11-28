"""
Simplified inference engine that works WITHOUT PyTorch
Uses lightweight libraries for basic emotion and sarcasm detection
"""
import io
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(audio_data, sr):
    """Extract basic acoustic features"""
    features = {}
    
    try:
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features['pitch_mean'] = float(pitch_mean)
        
        # Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = float(np.mean(rms))
        
        # Speaking rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['speaking_rate'] = float(np.mean(zcr))
        
    except Exception as e:
        print(f"Warning: Error extracting features: {e}")
    
    return features

def simple_transcribe(audio_data, sr):
    """Simple transcription using speech_recognition"""
    try:
        import speech_recognition as sr_lib
        import soundfile as sf
        import tempfile
        import os
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sr)
        
        try:
            recognizer = sr_lib.Recognizer()
            with sr_lib.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Could not transcribe audio"

def simple_emotion_detection(text, audio_features):
    """Simple rule-based emotion detection"""
    from textblob import TextBlob
    
    # Analyze sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Use audio features
    pitch = audio_features.get('pitch_mean', 0)
    energy = audio_features.get('energy_mean', 0)
    
    # Simple rules
    if polarity > 0.3:
        if energy > 0.05:
            return "happy", 0.75
        else:
            return "neutral", 0.60
    elif polarity < -0.3:
        if pitch > 200:
            return "angry", 0.70
        else:
            return "sad", 0.70
    elif energy > 0.08:
        return "surprise", 0.65
    else:
        return "neutral", 0.60

def simple_sarcasm_detection(text, audio_features):
    """Simple sarcasm detection"""
    from textblob import TextBlob
    
    sarcasm_score = 0.0
    indicators = []
    
    # Text analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Sarcastic phrases
    sarcasm_phrases = [
        'oh great', 'oh wonderful', 'just what i needed', 'yeah right',
        'sure', 'of course', 'obviously', 'clearly', 'perfect'
    ]
    if any(phrase in text.lower() for phrase in sarcasm_phrases):
        sarcasm_score += 0.5
        indicators.append("sarcastic_phrase")
    
    # Exaggeration
    exaggeration_words = ['absolutely', 'totally', 'completely', 'amazing', 'fantastic']
    if any(word in text.lower() for word in exaggeration_words) and polarity > 0:
        sarcasm_score += 0.3
        indicators.append("exaggeration")
    
    # High pitch
    if audio_features.get('pitch_mean', 0) > 200:
        sarcasm_score += 0.2
        indicators.append("high_pitch")
    
    sarcasm_score = min(sarcasm_score, 1.0)
    is_sarcastic = sarcasm_score > 0.5
    
    return is_sarcastic, sarcasm_score, indicators

def run_inference(audio_bytes: bytes, sample_rate: int = None):
    """
    Simplified inference without PyTorch
    """
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED INFERENCE (No PyTorch)")
    print(f"Processing audio: {len(audio_bytes)} bytes")
    
    try:
        # Load audio
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        print(f"Audio loaded: {len(audio_data)} samples, {sr}Hz, {len(audio_data)/sr:.2f}s")
        
        # Ensure minimum length
        min_samples = int(0.5 * sr)
        if len(audio_data) < min_samples:
            audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant')
        
        # 1. Transcribe
        print("Step 1/4: Transcribing audio...")
        transcript = simple_transcribe(audio_data, sr)
        print(f"✓ Transcript: '{transcript}'")
        
        if not transcript or len(transcript) < 3:
            transcript = "Unable to transcribe audio"
            return "neutral", 0.5, False, 0.0, transcript
        
        # 2. Extract features
        print("Step 2/4: Extracting audio features...")
        audio_features = extract_audio_features(audio_data, sr)
        print(f"✓ Features extracted")
        
        # 3. Detect emotion
        print("Step 3/4: Analyzing emotion...")
        emotion, confidence = simple_emotion_detection(transcript, audio_features)
        print(f"✓ Emotion: {emotion} ({confidence:.2f})")
        
        # 4. Detect sarcasm
        print("Step 4/4: Detecting sarcasm...")
        is_sarcastic, sarcasm_score, indicators = simple_sarcasm_detection(transcript, audio_features)
        print(f"✓ Sarcasm: {'YES' if is_sarcastic else 'NO'} ({sarcasm_score:.2f})")
        if indicators:
            print(f"  Indicators: {', '.join(indicators)}")
        
        print(f"{'='*60}\n")
        
        return emotion, round(confidence, 2), is_sarcastic, round(sarcasm_score, 2), transcript
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return "neutral", 0.5, False, 0.0, "Error processing audio"

print("✓ Simplified inference engine initialized (No PyTorch required)")

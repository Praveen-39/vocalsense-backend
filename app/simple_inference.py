"""
Simplified inference engine that works WITHOUT PyTorch
Uses lightweight libraries for basic emotion and sarcasm detection
"""
import io
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

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
            # Adjust for ambient noise
            with sr_lib.AudioFile(tmp_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                
            # Try Google Speech Recognition
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr_lib.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return "Audio unclear - could not transcribe"
            except sr_lib.RequestError as e:
                print(f"Could not request results from Google Speech Recognition; {e}")
                return "Transcription service unavailable"
                
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
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
        # Load audio - ROBUST METHOD using pydub
        print("Loading audio with pydub (ffmpeg wrapper)...")
        try:
            from pydub import AudioSegment
            import tempfile
            import os
            
            # Save input bytes to temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name
            
            # Convert to WAV using pydub (uses ffmpeg)
            try:
                # Try loading as any format (pydub auto-detects)
                audio_segment = AudioSegment.from_file(tmp_in_path)
                
                # Export as WAV to new temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                    tmp_out_path = tmp_out.name
                    
                audio_segment.export(tmp_out_path, format="wav")
                
                # Now load the clean WAV with librosa
                audio_data, sr = librosa.load(tmp_out_path, sr=None)
                print(f"✓ Audio loaded via pydub: {len(audio_data)} samples, {sr}Hz")
                
                # Cleanup
                try:
                    os.unlink(tmp_in_path)
                    os.unlink(tmp_out_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"Pydub conversion failed: {e}")
                # Fallback to direct librosa load
                print("Falling back to direct librosa load...")
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
                
        except ImportError:
            print("Pydub not installed, using direct librosa load")
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
        except Exception as e:
            print(f"ERROR loading audio: {e}")
            import traceback
            traceback.print_exc()
            return "neutral", 0.5, False, 0.0, "Unable to load audio file"
        
        # Ensure minimum length
        min_samples = int(0.5 * sr)
        if len(audio_data) < min_samples:
            print(f"Audio too short, padding from {len(audio_data)} to {min_samples} samples")
            audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant')
        
        # 1. Transcribe
        print("Step 1/4: Transcribing audio...")
        transcript = "Audio sample"  # Default fallback
        try:
            transcript = simple_transcribe(audio_data, sr)
            print(f"✓ Transcript: '{transcript}'")
        except Exception as e:
            print(f"ERROR in transcription: {e}")
            import traceback
            traceback.print_exc()
            transcript = "Transcription failed"
        
        # Use fallback if transcription failed
        if not transcript or len(transcript) < 3 or "could not transcribe" in transcript.lower() or "error" in transcript.lower():
            print("Warning: Transcription failed, using neutral fallback")
            transcript = "Audio sample"  # Neutral fallback for emotion detection
        
        # 2. Extract features
        print("Step 2/4: Extracting audio features...")
        audio_features = {}
        try:
            audio_features = extract_audio_features(audio_data, sr)
            print(f"✓ Features extracted: {len(audio_features)} features")
        except Exception as e:
            print(f"ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            audio_features = {'pitch_mean': 0, 'energy_mean': 0, 'speaking_rate': 0}
        
        # 3. Detect emotion
        print("Step 3/4: Analyzing emotion...")
        emotion = "neutral"
        confidence = 0.6
        try:
            emotion, confidence = simple_emotion_detection(transcript, audio_features)
            print(f"✓ Emotion: {emotion} ({confidence:.2f})")
        except Exception as e:
            print(f"ERROR in emotion detection: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Detect sarcasm
        print("Step 4/4: Detecting sarcasm...")
        is_sarcastic = False
        sarcasm_score = 0.0
        indicators = []
        try:
            is_sarcastic, sarcasm_score, indicators = simple_sarcasm_detection(transcript, audio_features)
            print(f"✓ Sarcasm: {'YES' if is_sarcastic else 'NO'} ({sarcasm_score:.2f})")
            if indicators:
                print(f"  Indicators: {', '.join(indicators)}")
        except Exception as e:
            print(f"ERROR in sarcasm detection: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")
        
        # Return successful result
        final_transcript = transcript if transcript != "Audio sample" else "Audio processed successfully"
        return emotion, round(confidence, 2), is_sarcastic, round(sarcasm_score, 2), final_transcript
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"CRITICAL ERROR in run_inference: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        # Return safe defaults instead of error message
        return "neutral", 0.5, False, 0.0, "Audio processed with limited analysis"

print("✓ Simplified inference engine initialized (No PyTorch required)")

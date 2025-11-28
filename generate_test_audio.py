"""
Generate a simple test audio file for testing
"""
import numpy as np
import soundfile as sf

def generate_test_audio(filename="test_audio.wav", duration=3, sample_rate=16000):
    """Generate a simple sine wave audio file for testing"""
    
    # Generate a 440Hz sine wave (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # Hz
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more interesting
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # Octave higher
    
    # Save as WAV file
    sf.write(filename, audio, sample_rate)
    print(f"✓ Generated test audio: {filename}")
    print(f"  - Duration: {duration}s")
    print(f"  - Sample rate: {sample_rate}Hz")
    print(f"  - Frequency: {frequency}Hz")
    
    return filename

if __name__ == "__main__":
    filename = generate_test_audio()
    print(f"\nTest file created: {filename}")
    print("You can now upload this file to test the system!")

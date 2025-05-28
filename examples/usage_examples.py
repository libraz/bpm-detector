"""Usage examples for BPM and Key detector."""

import os
import numpy as np
from bmp_detector import AudioAnalyzer, BPMDetector, KeyDetector


def basic_usage_example():
    """Basic usage example with AudioAnalyzer."""
    print("=== Basic Usage Example ===")
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Example audio file path (replace with your own)
    audio_file = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")
        return
    
    # Analyze BPM only
    print("Analyzing BPM only...")
    results = analyzer.analyze_file(audio_file, detect_key=False)
    print(f"BPM: {results['bpm']:.2f} (confidence: {results['bpm_confidence']:.1f}%)")
    
    # Analyze both BPM and key
    print("\nAnalyzing BPM and Key...")
    results = analyzer.analyze_file(audio_file, detect_key=True)
    print(f"BPM: {results['bpm']:.2f} (confidence: {results['bpm_confidence']:.1f}%)")
    print(f"Key: {results['key']} (confidence: {results['key_confidence']:.1f}%)")


def advanced_usage_example():
    """Advanced usage with separate detectors."""
    print("\n=== Advanced Usage Example ===")
    
    # Initialize separate detectors
    bpm_detector = BPMDetector(sr=44100, hop_length=512)
    key_detector = KeyDetector(hop_length=512)
    
    # Example: Generate synthetic audio for testing
    print("Generating synthetic audio for testing...")
    duration = 5.0  # seconds
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate a simple sine wave at 440 Hz (A4)
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some rhythm (simple beat pattern)
    beat_freq = 2.0  # 120 BPM = 2 beats per second
    beat_pattern = np.sin(2 * np.pi * beat_freq * t) > 0
    audio = audio * (0.5 + 0.5 * beat_pattern)
    
    print("Detecting BPM from synthetic audio...")
    bpm, confidence, candidates, votes = bpm_detector.detect(
        audio, sr, min_bpm=60, max_bpm=180
    )
    print(f"Detected BPM: {bpm:.2f} (confidence: {confidence:.1f}%)")
    
    print("Detecting key from synthetic audio...")
    key, key_confidence = key_detector.detect(audio, sr)
    print(f"Detected Key: {key} (confidence: {key_confidence:.1f}%)")


def batch_processing_example():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    # List of audio files (replace with your own)
    audio_files = [
        "path/to/audio1.wav",
        "path/to/audio2.mp3",
        "path/to/audio3.flac"
    ]
    
    analyzer = AudioAnalyzer()
    results = []
    
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"Skipping missing file: {audio_file}")
            continue
        
        print(f"Processing: {os.path.basename(audio_file)}")
        
        try:
            result = analyzer.analyze_file(audio_file, detect_key=True)
            results.append(result)
            
            print(f"  BPM: {result['bpm']:.2f}")
            print(f"  Key: {result['key']}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if results:
        avg_bpm = np.mean([r['bpm'] for r in results])
        print(f"\nProcessed {len(results)} files")
        print(f"Average BPM: {avg_bpm:.2f}")


def custom_parameters_example():
    """Example with custom detection parameters."""
    print("\n=== Custom Parameters Example ===")
    
    # Custom analyzer with specific parameters
    analyzer = AudioAnalyzer(sr=22050, hop_length=256)
    
    audio_file = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    
    # Custom BPM range for electronic music
    print("Analyzing with custom parameters for electronic music...")
    results = analyzer.analyze_file(
        audio_file,
        detect_key=True,
        min_bpm=80,    # Lower bound for electronic music
        max_bpm=200,   # Upper bound for electronic music
        start_bpm=128  # Common electronic music tempo
    )
    
    print(f"BPM: {results['bpm']:.2f}")
    print(f"Key: {results['key']}")
    
    # Show top candidates
    print("Top BPM candidates:")
    for bpm, votes in results['bpm_candidates'][:5]:
        print(f"  {bpm:6.2f} BPM : {votes} votes")


if __name__ == "__main__":
    print("BPM and Key Detector - Usage Examples")
    print("=" * 40)
    
    # Run examples
    basic_usage_example()
    advanced_usage_example()
    batch_processing_example()
    custom_parameters_example()
    
    print("\nFor more information, see the documentation at:")
    print("https://github.com/libraz/bpm-detector")
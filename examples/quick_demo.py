"""Quick demonstration of music analysis features with shorter audio."""

import os
import sys

import numpy as np
import soundfile as sf

# Add the src directory to the path so we can import bpm_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector import AudioAnalyzer


def create_short_test_audio():
    """Create a short test audio signal for quick demonstration."""
    print("Creating short test audio signal...")

    # Parameters - much shorter for quick demo
    duration = 5.0  # 5 seconds instead of 30
    sr = 22050

    # Generate time array
    t = np.linspace(0, duration, int(sr * duration))

    # Create a simple chord progression: C - Am - F - G
    chord_freqs = [
        [261.63, 329.63, 392.00],  # C major
        [220.00, 261.63, 329.63],  # A minor
        [174.61, 220.00, 261.63],  # F major
        [196.00, 246.94, 293.66],  # G major
    ]

    # Create signal with chord progression
    signal = np.zeros_like(t)
    chord_duration = duration / 4  # Each chord lasts 1/4 of the song

    for i, chord in enumerate(chord_freqs):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration

        # Find time indices for this chord
        chord_mask = (t >= start_time) & (t < end_time)
        chord_t = t[chord_mask]

        # Generate chord tones
        chord_signal = np.zeros_like(chord_t)
        for freq in chord:
            chord_signal += 0.3 * np.sin(2 * np.pi * freq * chord_t)

        # Add some rhythm (simple beat pattern)
        beat_freq = 2.0  # 120 BPM = 2 beats per second
        beat_pattern = (np.sin(2 * np.pi * beat_freq * chord_t) > 0).astype(float)
        chord_signal *= 0.5 + 0.5 * beat_pattern

        signal[chord_mask] = chord_signal

    # Add some noise for realism
    signal += 0.05 * np.random.randn(len(signal))

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal, sr


def quick_demo():
    """Quick demonstration of music analysis."""
    print("=== Quick Music Analysis Demo ===\n")

    # Create short test audio
    audio_signal, sr = create_short_test_audio()

    # Initialize analyzer
    print("Initializing AudioAnalyzer...")
    analyzer = AudioAnalyzer(sr=sr)

    # Save test audio temporarily
    test_file = "quick_test_audio.wav"
    sf.write(test_file, audio_signal, sr)

    try:
        print(f"Analyzing short test audio ({len(audio_signal)/sr:.1f} seconds)...\n")

        # First, basic analysis only (fast)
        print("--- BASIC ANALYSIS (Fast) ---")
        import time

        start_time = time.time()

        basic_results = analyzer.analyze_file(test_file, detect_key=True, comprehensive=False)  # Basic only

        basic_time = time.time() - start_time
        print(f"Basic analysis completed in {basic_time:.2f} seconds")

        # Display basic results
        basic_info = basic_results.get('basic_info', {})
        print(f"BPM: {basic_info.get('bpm', 0):.1f}")
        print(f"Key: {basic_info.get('key', 'Unknown')}")
        print(f"Duration: {basic_info.get('duration', 0):.1f} seconds\n")

        # Now comprehensive analysis
        print("--- COMPREHENSIVE ANALYSIS (Slower) ---")
        start_time = time.time()

        comprehensive_results = analyzer.analyze_file(test_file, detect_key=True, comprehensive=True)

        comprehensive_time = time.time() - start_time
        print(f"Comprehensive analysis completed in {comprehensive_time:.2f} seconds")
        print(f"Speed difference: {comprehensive_time/basic_time:.1f}x slower\n")

        # Display some comprehensive results
        chord_prog = comprehensive_results.get('chord_progression', {})
        structure = comprehensive_results.get('structure', {})
        rhythm = comprehensive_results.get('rhythm', {})

        print("🎹 CHORD PROGRESSION:")
        main_prog = chord_prog.get('main_progression', [])
        if main_prog:
            print(f"   Main progression: {' → '.join(main_prog)}")
        print(f"   Chord complexity: {chord_prog.get('chord_complexity', 0):.1%}")

        print("\n🏗️ STRUCTURE:")
        print(f"   Sections: {structure.get('section_count', 0)}")
        print(f"   Form: {structure.get('form', 'Unknown')}")

        print("\n🥁 RHYTHM:")
        print(f"   Time signature: {rhythm.get('time_signature', '4/4')}")
        print(f"   Groove type: {rhythm.get('groove_type', 'straight')}")

        # Generate reference sheet
        print("\n📋 REFERENCE SHEET:")
        print("-" * 40)
        reference_sheet = analyzer.generate_reference_sheet(comprehensive_results)
        # Show first few lines
        lines = reference_sheet.split('\n')[:15]
        print('\n'.join(lines))
        if len(reference_sheet.split('\n')) > 15:
            print("... (truncated)")

        print(f"\n✅ Demo completed successfully!")
        print(f"💡 Tip: Use comprehensive=False for faster analysis when you only need BPM/Key")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🧹 Cleaned up temporary file: {test_file}")


if __name__ == "__main__":
    quick_demo()

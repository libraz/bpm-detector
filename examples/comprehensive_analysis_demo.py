"""Demonstration of comprehensive music analysis features."""

import os
import sys

import numpy as np

# Add the src directory to the path so we can import bpm_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector import AudioAnalyzer


def create_test_audio():
    """Create a simple test audio signal for demonstration."""
    print("Creating test audio signal...")

    # Parameters
    duration = 30.0  # 30 seconds
    sr = 22050

    # Generate time array
    t = np.linspace(0, duration, int(sr * duration))

    # Create a simple musical signal
    # Base frequency (A4 = 440 Hz)

    # Create a simple chord progression: A - F#m - D - E
    chord_freqs = [
        [440, 554.37, 659.25],  # A major
        [369.99, 440, 554.37],  # F# minor
        [293.66, 369.99, 440],  # D major
        [329.63, 415.30, 493.88],  # E major
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


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive music analysis."""
    print("=== Comprehensive Music Analysis Demo ===\n")

    # Create test audio
    audio_signal, sr = create_test_audio()

    # Initialize analyzer
    print("Initializing AudioAnalyzer...")
    analyzer = AudioAnalyzer(sr=sr)

    # Save test audio temporarily
    test_file = "temp_test_audio.wav"
    import soundfile as sf

    sf.write(test_file, audio_signal, sr)

    try:
        print(f"Analyzing test audio ({len(audio_signal)/sr:.1f} seconds)...\n")

        # Perform comprehensive analysis
        results = analyzer.analyze_file(test_file, detect_key=True, comprehensive=True)

        # Display results
        print("--- ANALYSIS RESULTS ---\n")

        # Basic information
        basic_info = results.get('basic_info', {})
        print("🎵 BASIC INFORMATION:")
        print(f"   BPM: {basic_info.get('bpm', 0):.1f}")
        print(f"   Key: {basic_info.get('key', 'Unknown')}")
        print(f"   Duration: {basic_info.get('duration', 0):.1f} seconds")
        print(f"   BPM Confidence: {basic_info.get('bpm_confidence', 0):.1f}%\n")

        # Chord progression
        chord_prog = results.get('chord_progression', {})
        print("🎹 CHORD PROGRESSION:")
        main_prog = chord_prog.get('main_progression', [])
        if main_prog:
            print(f"   Main progression: {' → '.join(main_prog)}")
        print(f"   Harmonic rhythm: {chord_prog.get('harmonic_rhythm', 0):.2f} changes/sec")
        print(f"   Chord complexity: {chord_prog.get('chord_complexity', 0):.1%}")
        print(f"   Unique chords: {chord_prog.get('unique_chords', 0)}\n")

        # Structure
        structure = results.get('structure', {})
        print("🏗️ STRUCTURE:")
        print(f"   Form: {structure.get('form', 'Unknown')}")
        print(f"   Sections: {structure.get('section_count', 0)}")
        print(f"   Repetition ratio: {structure.get('repetition_ratio', 0):.1%}")
        print(f"   Structural complexity: {structure.get('structural_complexity', 0):.1%}")

        sections = structure.get('sections', [])
        if sections:
            print("   Section breakdown:")
            for section in sections:
                print(
                    f"     {section.get('type', 'unknown')}: "
                    f"{section.get('start_time', 0):.1f}s - "
                    f"{section.get('end_time', 0):.1f}s "
                    f"({section.get('duration', 0):.1f}s)"
                )
        print()

        # Rhythm
        rhythm = results.get('rhythm', {})
        print("🥁 RHYTHM:")
        print(f"   Time signature: {rhythm.get('time_signature', '4/4')}")
        print(f"   Groove type: {rhythm.get('groove_type', 'straight')}")
        print(f"   Syncopation level: {rhythm.get('syncopation_level', 0):.1%}")
        print(f"   Rhythmic complexity: {rhythm.get('rhythmic_complexity', 0):.1%}")
        print(f"   Swing ratio: {rhythm.get('swing_ratio', 0.5):.2f}\n")

        # Timbre
        timbre = results.get('timbre', {})
        print("🎨 TIMBRE:")
        print(f"   Brightness: {timbre.get('brightness', 0):.1%}")
        print(f"   Warmth: {timbre.get('warmth', 0):.1%}")
        print(f"   Roughness: {timbre.get('roughness', 0):.1%}")
        print(f"   Density: {timbre.get('density', 0):.1%}")

        instruments = timbre.get('dominant_instruments', [])
        if instruments:
            print("   Detected instruments:")
            for inst in instruments[:3]:
                print(f"     {inst.get('instrument', 'unknown')}: " f"{inst.get('confidence', 0):.1%} confidence")
        print()

        # Melody & Harmony
        melody_harmony = results.get('melody_harmony', {})
        print("🎼 MELODY & HARMONY:")

        melodic_range = melody_harmony.get('melodic_range', {})
        print(f"   Melodic range: {melodic_range.get('range_octaves', 0):.1f} octaves")

        consonance = melody_harmony.get('consonance', {})
        print(f"   Consonance level: {consonance.get('consonance_level', 0):.1%}")

        harmony_complexity = melody_harmony.get('harmony_complexity', {})
        print(f"   Harmonic complexity: {harmony_complexity.get('harmonic_complexity', 0):.1%}")
        print(f"   Melody present: {melody_harmony.get('melody_present', False)}\n")

        # Dynamics
        dynamics = results.get('dynamics', {})
        print("📊 DYNAMICS:")

        dynamic_range = dynamics.get('dynamic_range', {})
        print(f"   Dynamic range: {dynamic_range.get('dynamic_range_db', 0):.1f} dB")

        loudness = dynamics.get('loudness', {})
        print(f"   Average loudness: {loudness.get('average_loudness_db', -30):.1f} dB")
        print(f"   Overall energy: {dynamics.get('overall_energy', 0):.1%}")

        climax_points = dynamics.get('climax_points', [])
        if climax_points:
            print(f"   Climax points: {len(climax_points)}")
            for i, climax in enumerate(climax_points[:2]):
                print(f"     {i+1}. {climax.get('time', 0):.1f}s " f"(intensity: {climax.get('intensity', 0):.1%})")
        print()

        # Reference tags
        tags = results.get('reference_tags', [])
        if tags:
            print("🏷️ REFERENCE TAGS:")
            print(f"   {', '.join(tags)}\n")

        # Generate reference sheet
        print("📋 REFERENCE SHEET:")
        print("-" * 40)
        reference_sheet = analyzer.generate_reference_sheet(results)
        print(reference_sheet)

        # Feature vector info
        similarity_features = results.get('similarity_features', {})
        feature_vector = similarity_features.get('feature_vector', [])
        if feature_vector:
            print("\n🔢 FEATURE VECTOR:")
            print(f"   Length: {len(feature_vector)} features")
            print(f"   Sample values: {feature_vector[:5]}")

        print("\n✅ Analysis completed successfully!")

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
    demonstrate_comprehensive_analysis()

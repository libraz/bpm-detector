#!/usr/bin/env python3
"""
Selective Analysis Demo

This example demonstrates how to use selective analysis to only compute
the features you need, resulting in faster processing times.

Usage:
    python selective_analysis_demo.py path/to/audio.wav
"""

import sys
from pathlib import Path

# Add parent directory to path to import bpm_detector
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bpm_detector import AudioAnalyzer


def demo_basic_only():
    """Demo 1: Basic BPM only (fastest)"""
    print("\n" + "=" * 60)
    print("Demo 1: Basic BPM Only (Fastest)")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Just BPM, no additional analysis
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=False,
        comprehensive=False
    )

    print(f"BPM: {results['basic_info']['bpm']:.1f}")
    print(f"Confidence: {results['basic_info']['bpm_confidence']:.1f}%")
    print(f"Duration: {results['basic_info']['duration']:.1f}s")


def demo_rhythm_only():
    """Demo 2: BPM + Rhythm/Time Signature"""
    print("\n" + "=" * 60)
    print("Demo 2: BPM + Rhythm Analysis")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Analyze rhythm and time signature
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=False,
        comprehensive=False,
        analyze_rhythm=True
    )

    basic = results['basic_info']
    rhythm = results['rhythm']

    print(f"BPM: {basic['bpm']:.1f}")
    print(f"Time Signature: {rhythm['time_signature']}")
    print(f"Groove Type: {rhythm['groove_type']}")
    print(f"Syncopation Level: {rhythm['syncopation_level']:.2f}")


def demo_key_and_rhythm():
    """Demo 3: BPM + Key + Rhythm"""
    print("\n" + "=" * 60)
    print("Demo 3: BPM + Key + Rhythm Analysis")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Most common combination for music cataloging
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=True,
        comprehensive=False,
        analyze_rhythm=True
    )

    basic = results['basic_info']
    rhythm = results['rhythm']

    print(f"BPM: {basic['bpm']:.1f}")
    print(f"Key: {basic['key']}")
    print(f"Time Signature: {rhythm['time_signature']}")
    print(f"Groove: {rhythm['groove_type']}")


def demo_production_focused():
    """Demo 4: Production-focused analysis (timbre + melody + dynamics)"""
    print("\n" + "=" * 60)
    print("Demo 4: Production-Focused Analysis")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Useful for mixing and production decisions
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=True,
        comprehensive=False,
        analyze_timbre=True,
        analyze_melody=True,
        analyze_dynamics=True
    )

    basic = results['basic_info']
    timbre = results['timbre']
    melody = results['melody_harmony']
    dynamics = results['dynamics']

    print(f"BPM: {basic['bpm']:.1f}, Key: {basic['key']}")
    print(f"\nTimbre:")
    print(f"  Brightness: {timbre['brightness']:.2f}")
    print(f"  Warmth: {timbre['warmth']:.2f}")
    print(f"  Dominant Instruments: {[inst['instrument'] for inst in timbre['dominant_instruments'][:3]]}")

    print(f"\nMelody:")
    if melody.get('melody_present'):
        melodic_range = melody['melodic_range']
        print(f"  Range: {melodic_range['lowest_note_name']} - {melodic_range['highest_note_name']}")
        print(f"  Range Category: {melodic_range['vocal_range_category']}")

    print(f"\nDynamics:")
    print(f"  Dynamic Range: {dynamics['dynamic_range']['dynamic_range_db']:.1f} dB")
    print(f"  Overall Energy: {dynamics['overall_energy']:.2f}")


def demo_structure_analysis():
    """Demo 5: Structure-focused analysis"""
    print("\n" + "=" * 60)
    print("Demo 5: Structure Analysis")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Useful for understanding song arrangement
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=False,
        comprehensive=False,
        analyze_structure=True,
        analyze_chords=True
    )

    basic = results['basic_info']
    structure = results['structure']
    chords = results['chord_progression']

    print(f"BPM: {basic['bpm']:.1f}")
    print(f"\nStructure:")
    print(f"  Form: {structure.get('form', 'N/A')}")
    print(f"  Sections: {structure.get('section_count', 0)}")

    if 'sections' in structure:
        print(f"  Section List:")
        for i, section in enumerate(structure['sections'][:5], 1):
            print(f"    {i}. {section['type']} ({section['start_time']:.1f}s - {section['duration']:.1f}s)")

    print(f"\nChord Progression:")
    if 'main_progression' in chords:
        print(f"  Main: {' → '.join(chords['main_progression'][:8])}")
    print(f"  Complexity: {chords.get('chord_complexity', 0):.1%}")


def demo_custom_combination():
    """Demo 6: Custom combination of analyses"""
    print("\n" + "=" * 60)
    print("Demo 6: Custom Analysis Combination")
    print("=" * 60)

    analyzer = AudioAnalyzer()

    # Mix and match based on your needs
    results = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=True,
        comprehensive=False,
        analyze_rhythm=True,
        analyze_chords=True,
        analyze_melody=True
    )

    print("Custom analysis with:")
    print("  ✓ BPM")
    print("  ✓ Key Detection")
    print("  ✓ Rhythm Analysis")
    print("  ✓ Chord Progression")
    print("  ✓ Melody & Harmony")
    print("\nSkipped (for faster processing):")
    print("  ✗ Structure Analysis")
    print("  ✗ Timbre Analysis")
    print("  ✗ Dynamics Analysis")

    basic = results['basic_info']
    rhythm = results['rhythm']
    chords = results['chord_progression']

    print(f"\nResults:")
    print(f"  BPM: {basic['bpm']:.1f}, Key: {basic['key']}")
    print(f"  Time: {rhythm['time_signature']}")
    print(f"  Main Chords: {' → '.join(chords['main_progression'][:4])}")


def demo_performance_comparison():
    """Demo 7: Performance comparison between selective and comprehensive"""
    print("\n" + "=" * 60)
    print("Demo 7: Performance Comparison")
    print("=" * 60)

    import time

    analyzer = AudioAnalyzer()

    # Selective analysis (rhythm only)
    start = time.time()
    results_selective = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=False,
        comprehensive=False,
        analyze_rhythm=True
    )
    selective_time = time.time() - start

    # Comprehensive analysis (all features)
    start = time.time()
    results_comprehensive = analyzer.analyze_file(
        'test_audio.wav',
        detect_key=True,
        comprehensive=True
    )
    comprehensive_time = time.time() - start

    print(f"Selective Analysis (BPM + Rhythm): {selective_time:.2f}s")
    print(f"Comprehensive Analysis (All Features): {comprehensive_time:.2f}s")
    print(f"Speedup: {comprehensive_time / selective_time:.1f}x faster")

    print(f"\nBoth analyses returned:")
    print(f"  BPM: {results_selective['basic_info']['bpm']:.1f}")
    print(f"  Time Signature: {results_selective['rhythm']['time_signature']}")


def main():
    """Run all demos"""
    if len(sys.argv) > 1:
        # Use provided audio file
        audio_file = sys.argv[1]
        print(f"Using audio file: {audio_file}")

        # Update demo functions to use the provided file
        # For simplicity, we'll just show the API usage

    else:
        # Show API usage without actual files
        print("\n" + "=" * 60)
        print("Selective Analysis Demo - API Usage Examples")
        print("=" * 60)
        print("\nTo run with an actual audio file:")
        print("  python selective_analysis_demo.py path/to/audio.wav")

        print("\n\nAPI Usage Examples:")
        print("\n1. BPM Only (fastest):")
        print("   results = analyzer.analyze_file('audio.wav', detect_key=False, comprehensive=False)")

        print("\n2. BPM + Rhythm:")
        print("   results = analyzer.analyze_file('audio.wav', comprehensive=False, analyze_rhythm=True)")

        print("\n3. BPM + Key + Rhythm (common use case):")
        print("   results = analyzer.analyze_file('audio.wav', detect_key=True, analyze_rhythm=True)")

        print("\n4. Production-focused (timbre + melody + dynamics):")
        print("   results = analyzer.analyze_file('audio.wav', detect_key=True,")
        print("                                    analyze_timbre=True, analyze_melody=True, analyze_dynamics=True)")

        print("\n5. Structure-focused (structure + chords):")
        print("   results = analyzer.analyze_file('audio.wav', analyze_structure=True, analyze_chords=True)")

        print("\n6. Custom combination:")
        print("   results = analyzer.analyze_file('audio.wav', detect_key=True,")
        print("                                    analyze_rhythm=True, analyze_chords=True, analyze_melody=True)")

        print("\n\nAvailable Parameters:")
        print("  detect_key=True/False     - Enable key detection")
        print("  comprehensive=True/False  - Enable all features (overrides individual flags)")
        print("  analyze_rhythm=True       - Analyze rhythm and time signature")
        print("  analyze_chords=True       - Analyze chord progressions")
        print("  analyze_structure=True    - Analyze musical structure")
        print("  analyze_timbre=True       - Analyze timbre and instruments")
        print("  analyze_melody=True       - Analyze melody and harmony")
        print("  analyze_dynamics=True     - Analyze dynamics")


if __name__ == '__main__':
    main()

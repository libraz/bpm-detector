#!/usr/bin/env python3
"""
Test for improved analysis features
Verification of fixes for issues pointed out in review
"""

import os
import sys
import time
from typing import Dict

# Add the src directory to the path so we can import bpm_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector.music_analyzer import AudioAnalyzer  # noqa: E402


def test_improved_analysis():
    """Test improved analysis features"""

    # Target audio file
    audio_file = "examples/icecream.mp3"

    if not os.path.exists(audio_file):
        print(f"❌ Test file not found: {audio_file}")
        return

    print("🎧 Testing Improved Analysis Features")
    print("=" * 60)
    print(f"📁 File: {audio_file}")
    print()

    # Initialize analyzer
    analyzer = AudioAnalyzer()

    # Execute analysis
    print("🔍 Starting analysis...")
    start_time = time.time()

    try:
        results = analyzer.analyze_file(audio_file)
        analysis_time = time.time() - start_time

        print(f"✅ Analysis completed ({analysis_time:.2f} seconds)")
        print()

        # Verify Key Detection improvements
        print("🎵 Key Detection Improvement Results:")
        print("-" * 40)
        key_info = results.get('melody_harmony', {}).get('key_detection', {})
        print(f"Key: {key_info.get('key', 'None')}")
        print(f"Mode: {key_info.get('mode', 'Unknown')}")
        print(f"Confidence: {key_info.get('confidence', 0.0):.3f}")
        print(f"Key Strength: {key_info.get('key_strength', 0.0):.3f}")
        print()

        # Verify structure analysis improvements
        print("🏗️ Structure Analysis Improvement Results:")
        print("-" * 40)
        structure = results.get('structure', {})
        sections = structure.get('sections', [])

        print(f"Total sections: {len(sections)}")
        print(f"Form: {structure.get('form', 'Unknown')}")
        print()

        # Check section type distribution
        section_types: Dict[str, int] = {}
        for section in sections:
            section_type = section.get('type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1

        print("Section type distribution:")
        for section_type, count in sorted(section_types.items()):
            print(f"  {section_type}: {count} sections")

        # Specifically check outro count
        outro_count = section_types.get('outro', 0)
        print(f"\n🎯 Outro section count: {outro_count}")
        if outro_count <= 2:
            print("✅ Excessive outro problem has been improved!")
        else:
            print("⚠️ Still too many outros")

        print()

        # Detailed section information
        print("📋 Detailed Section Information:")
        print("-" * 40)
        for i, section in enumerate(sections):
            print(
                f"{i+1:2d}. {section.get('type', 'unknown'):12s} "
                f"{section.get('start_time', 0):6.1f}s - {section.get('end_time', 0):6.1f}s "
                f"(Energy: {section.get('energy_level', 0):.2f}, "
                f"Complexity: {section.get('complexity', 0):.2f})"
            )

        print()

        # Verify harmony analysis improvements
        print("🎼 Harmony Analysis Improvement Results:")
        print("-" * 40)
        consonance = results.get('melody_harmony', {}).get('consonance', {})
        print(f"Consonance Level: {consonance.get('consonance_level', 0):.1%}")
        print(f"Dissonance Level: {consonance.get('dissonance_level', 0):.1%}")

        harmony_complexity = results.get('melody_harmony', {}).get('harmony_complexity', {})
        print(f"Harmonic Complexity: {harmony_complexity.get('harmonic_complexity', 0):.1%}")

        print()
        print("🎯 Improvement Verification:")
        print("-" * 40)

        # Verify Key Detection improvements
        if key_info.get('key') != 'None':
            print("✅ Key Detection: Key was detected")
        else:
            print("❌ Key Detection: Key still not detected")

        # Verify excessive outro problem fix
        if outro_count <= 2:
            print("✅ Section Classification: Excessive outro problem resolved")
        else:
            print("❌ Section Classification: Excessive outro problem persists")

        # Verify consonance validity
        consonance_level = consonance.get('consonance_level', 0)
        if 0.7 <= consonance_level <= 0.95:
            print("✅ Consonance Analysis: Valid value for pop song")
        else:
            print(f"⚠️ Consonance Analysis: Value out of range ({consonance_level:.1%})")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_improved_analysis()

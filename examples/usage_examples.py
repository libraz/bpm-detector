"""Usage examples for comprehensive music analysis."""

import json
import os

import numpy as np

from bpm_detector import (
    AudioAnalyzer,
    BPMDetector,
    ChordProgressionAnalyzer,
    DynamicsAnalyzer,
    KeyDetector,
    MelodyHarmonyAnalyzer,
    RhythmAnalyzer,
    SimilarityEngine,
    StructureAnalyzer,
    TimbreAnalyzer,
)


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
    bpm, confidence, candidates, votes = bpm_detector.detect(audio, sr, min_bpm=60, max_bpm=180)
    print(f"Detected BPM: {bpm:.2f} (confidence: {confidence:.1f}%)")

    print("Detecting key from synthetic audio...")
    key, key_confidence = key_detector.detect(audio, sr)
    print(f"Detected Key: {key} (confidence: {key_confidence:.1f}%)")


def batch_processing_example():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")

    # List of audio files (replace with your own)
    audio_files = ["path/to/audio1.wav", "path/to/audio2.mp3", "path/to/audio3.flac"]

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
        min_bpm=80,  # Lower bound for electronic music
        max_bpm=200,  # Upper bound for electronic music
        start_bpm=128,  # Common electronic music tempo
    )

    print(f"BPM: {results['bpm']:.2f}")
    print(f"Key: {results['key']}")

    # Show top candidates
    print("Top BPM candidates:")
    for bpm, votes in results['bpm_candidates'][:5]:
        print(f"  {bpm:6.2f} BPM : {votes} votes")


def comprehensive_analysis_example():
    """Example of comprehensive music analysis."""
    print("\n=== Comprehensive Analysis Example ===")

    # Initialize analyzer
    analyzer = AudioAnalyzer()

    # Example audio file path (replace with your own)
    audio_file = "path/to/your/audio.wav"

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")
        return

    print("Performing comprehensive analysis...")

    # Comprehensive analysis
    results = analyzer.analyze_file(audio_file, detect_key=True, comprehensive=True)

    # Display results
    print("\n--- Analysis Results ---")

    # Basic info
    basic_info = results.get('basic_info', {})
    print(f"BPM: {basic_info.get('bpm', 0):.2f}")
    print(f"Key: {basic_info.get('key', 'Unknown')}")
    print(f"Duration: {basic_info.get('duration', 0):.1f} seconds")

    # Chord progression
    chord_prog = results.get('chord_progression', {})
    main_prog = chord_prog.get('main_progression', [])
    if main_prog:
        print(f"Main chord progression: {' - '.join(main_prog)}")

    # Structure
    structure = results.get('structure', {})
    sections = structure.get('sections', [])
    print(f"Song structure: {len(sections)} sections")
    for section in sections[:3]:  # Show first 3 sections
        print(f"  {section.get('type', 'unknown')}: {section.get('duration', 0):.1f}s")

    # Rhythm
    rhythm = results.get('rhythm', {})
    print(f"Time signature: {rhythm.get('time_signature', '4/4')}")
    print(f"Groove: {rhythm.get('groove_type', 'straight')}")

    # Timbre
    timbre = results.get('timbre', {})
    instruments = timbre.get('dominant_instruments', [])
    if instruments:
        print("Main instruments:")
        for inst in instruments[:3]:
            print(f"  {inst.get('instrument', 'unknown')}: {inst.get('confidence', 0):.1%}")

    # Generate reference sheet
    print("\n--- Reference Sheet ---")
    reference_sheet = analyzer.generate_reference_sheet(results)
    print(reference_sheet[:500] + "..." if len(reference_sheet) > 500 else reference_sheet)


def similarity_analysis_example():
    """Example of similarity analysis between tracks."""
    print("\n=== Similarity Analysis Example ===")

    # Initialize components
    analyzer = AudioAnalyzer()
    similarity_engine = SimilarityEngine()

    # Example audio files (replace with your own)
    audio_files = ["path/to/track1.wav", "path/to/track2.wav", "path/to/track3.wav"]

    # Check if files exist
    existing_files = [f for f in audio_files if os.path.exists(f)]
    if len(existing_files) < 2:
        print("Need at least 2 audio files for similarity analysis")
        print("Please provide valid audio file paths.")
        return

    print(f"Analyzing {len(existing_files)} tracks for similarity...")

    # Analyze all tracks
    track_features = []
    track_names = []

    for audio_file in existing_files:
        print(f"Analyzing: {os.path.basename(audio_file)}")

        try:
            results = analyzer.analyze_file(audio_file, comprehensive=True)
            feature_vector = similarity_engine.extract_feature_vector(results)

            track_features.append(feature_vector)
            track_names.append(os.path.basename(audio_file))

        except Exception as e:
            print(f"Error analyzing {audio_file}: {e}")

    if len(track_features) < 2:
        print("Not enough tracks successfully analyzed")
        return

    # Calculate similarity matrix
    print("\nCalculating similarities...")

    similarity_matrix = similarity_engine.generate_similarity_matrix(track_features, method='weighted')

    # Display similarity results
    print("\n--- Similarity Matrix ---")
    print("Track similarities (0.0 = different, 1.0 = identical):")

    for i, name1 in enumerate(track_names):
        for j, name2 in enumerate(track_names):
            if i < j:  # Only show upper triangle
                similarity = similarity_matrix[i, j]
                print(f"{name1} <-> {name2}: {similarity:.3f}")


def individual_analyzer_example():
    """Example using individual analyzers."""
    print("\n=== Individual Analyzer Example ===")

    # Example audio file path (replace with your own)
    audio_file = "path/to/your/audio.wav"

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return

    # Load audio
    import librosa

    y, sr = librosa.load(audio_file, sr=22050, mono=True)

    print("Using individual analyzers...")

    # Chord analysis
    chord_analyzer = ChordProgressionAnalyzer()
    chord_results = chord_analyzer.analyze(y, sr, "C Major")
    print(f"Chord complexity: {chord_results.get('chord_complexity', 0):.2f}")

    # Rhythm analysis
    rhythm_analyzer = RhythmAnalyzer()
    rhythm_results = rhythm_analyzer.analyze(y, sr)
    print(f"Syncopation level: {rhythm_results.get('syncopation_level', 0):.2f}")

    # Timbre analysis
    timbre_analyzer = TimbreAnalyzer()
    timbre_results = timbre_analyzer.analyze(y, sr)
    print(f"Brightness: {timbre_results.get('brightness', 0):.2f}")

    # Structure analysis
    structure_analyzer = StructureAnalyzer()
    structure_results = structure_analyzer.analyze(y, sr)
    print(f"Structural complexity: {structure_results.get('structural_complexity', 0):.2f}")


def export_analysis_example():
    """Example of exporting analysis results."""
    print("\n=== Export Analysis Example ===")

    # Initialize analyzer
    analyzer = AudioAnalyzer()

    # Example audio file path (replace with your own)
    audio_file = "path/to/your/audio.wav"

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return

    print("Analyzing and exporting results...")

    # Perform analysis
    results = analyzer.analyze_file(audio_file, comprehensive=True)

    # Export to JSON
    output_file = "analysis_results.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    serializable_results = convert_numpy(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"Results exported to: {output_file}")

    # Generate and save reference sheet
    reference_sheet = analyzer.generate_reference_sheet(results)

    sheet_file = "reference_sheet.md"
    with open(sheet_file, 'w', encoding='utf-8') as f:
        f.write(reference_sheet)

    print(f"Reference sheet saved to: {sheet_file}")


if __name__ == "__main__":
    print("Comprehensive Music Analysis - Usage Examples")
    print("=" * 50)

    # Run examples
    basic_usage_example()
    advanced_usage_example()
    batch_processing_example()
    custom_parameters_example()
    comprehensive_analysis_example()
    similarity_analysis_example()
    individual_analyzer_example()
    export_analysis_example()

    print("\nFor more information, see the documentation at:")
    print("https://github.com/libraz/bpm-detector")

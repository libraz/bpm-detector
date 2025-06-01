"""Demonstration of parallel processing capabilities."""

import os
import tempfile
import time

import numpy as np
import soundfile as sf

from bpm_detector import AudioAnalyzer, AutoParallelConfig, SmartParallelAudioAnalyzer


def create_test_audio_files(num_files: int = 5, duration: float = 10.0) -> list:
    """Create test audio files for demonstration."""
    temp_dir = tempfile.mkdtemp()
    test_files = []

    print(f"Creating {num_files} test audio files ({duration}s each)...")

    for i in range(num_files):
        # Generate different musical content for each file
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))

        # Create a simple musical pattern with different characteristics
        base_freq = 220 + i * 55  # Different base frequencies

        # Main melody
        melody = np.sin(2 * np.pi * base_freq * t)

        # Add harmonics
        harmony = 0.3 * np.sin(2 * np.pi * base_freq * 1.5 * t)

        # Add rhythm (simple beat pattern)
        beat_freq = 2 + i * 0.5  # Different tempos
        rhythm = 0.2 * np.sin(2 * np.pi * beat_freq * t) * np.sin(2 * np.pi * base_freq * 2 * t)

        # Combine elements
        audio = melody + harmony + rhythm

        # Add some noise for realism
        noise = 0.05 * np.random.randn(len(audio))
        audio = audio + noise

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8

        # Save file
        filename = f"test_song_{i+1}_{int(base_freq)}hz.wav"
        filepath = os.path.join(temp_dir, filename)
        sf.write(filepath, audio, sr)
        test_files.append(filepath)

    print(f"Created test files in: {temp_dir}")
    return test_files


def demonstrate_system_info():
    """Demonstrate system information display."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    config = AutoParallelConfig.get_optimal_config()

    print("🖥️  System Configuration:")
    print(f"   Parallel Enabled: {config.enable_parallel}")
    print(f"   Max Workers: {config.max_workers}")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   Memory Limit: {config.memory_limit_mb} MB")
    print(f"   Reason: {config.reason}")
    print()


def demonstrate_single_file_analysis():
    """Demonstrate single file analysis with parallel processing."""
    print("=" * 60)
    print("SINGLE FILE ANALYSIS COMPARISON")
    print("=" * 60)

    # Create a test file
    test_files = create_test_audio_files(1, 15.0)  # 15 second file
    test_file = test_files[0]

    print(f"Analyzing: {os.path.basename(test_file)}")
    print()

    # Sequential analysis
    print("🔄 Sequential Analysis:")
    sequential_analyzer = AudioAnalyzer()

    start_time = time.time()
    sequential_results = sequential_analyzer.analyze_file(test_file, comprehensive=True)
    sequential_time = time.time() - start_time

    print(f"   Time: {sequential_time:.2f} seconds")
    print(f"   BPM: {sequential_results['basic_info']['bpm']:.1f}")
    print(f"   Key: {sequential_results['basic_info']['key']}")
    print()

    # Parallel analysis
    print("🚀 Parallel Analysis:")
    parallel_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def progress_callback(progress: float, message: str = ""):
        print(f"   Progress: {progress:.1f}% - {message}")

    start_time = time.time()
    parallel_results = parallel_analyzer.analyze_file(
        test_file, comprehensive=True, progress_callback=progress_callback
    )
    parallel_time = time.time() - start_time

    print(f"   Time: {parallel_time:.2f} seconds")
    print(f"   BPM: {parallel_results['basic_info']['bpm']:.1f}")
    print(f"   Key: {parallel_results['basic_info']['key']}")
    print()

    # Performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print("📊 Performance:")
    print(f"   Sequential: {sequential_time:.2f}s")
    print(f"   Parallel: {parallel_time:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print()

    # Cleanup
    for file in test_files:
        os.remove(file)
    os.rmdir(os.path.dirname(test_file))


def demonstrate_multiple_file_analysis():
    """Demonstrate multiple file analysis with parallel processing."""
    print("=" * 60)
    print("MULTIPLE FILE ANALYSIS COMPARISON")
    print("=" * 60)

    # Create test files
    test_files = create_test_audio_files(5, 8.0)  # 5 files, 8 seconds each

    print(f"Analyzing {len(test_files)} files:")
    for file in test_files:
        print(f"  - {os.path.basename(file)}")
    print()

    # Sequential analysis
    print("🔄 Sequential Analysis:")
    sequential_analyzer = AudioAnalyzer()

    start_time = time.time()
    sequential_results = {}
    for i, file in enumerate(test_files):
        print(f"   Processing file {i+1}/{len(test_files)}: {os.path.basename(file)}")
        sequential_results[file] = sequential_analyzer.analyze_file(file, comprehensive=True)
    sequential_time = time.time() - start_time

    print(f"   Total time: {sequential_time:.2f} seconds")
    print()

    # Parallel analysis
    print("🚀 Parallel Analysis:")
    parallel_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def multi_progress_callback(progress: float, message: str = ""):
        print(f"   Overall Progress: {progress:.1f}% - {message}")

    start_time = time.time()
    parallel_results = parallel_analyzer.analyze_file(
        test_files, comprehensive=True, progress_callback=multi_progress_callback
    )
    parallel_time = time.time() - start_time

    print(f"   Total time: {parallel_time:.2f} seconds")
    print()

    # Results comparison
    print("📊 Results Summary:")
    print(f"   Files processed: {len(parallel_results)}")

    for file in test_files:
        if file in parallel_results and 'basic_info' in parallel_results[file]:
            result = parallel_results[file]['basic_info']
            print(f"   {os.path.basename(file)}: {result['bpm']:.1f} BPM, {result['key']}")

    # Performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print()
    print("📈 Performance:")
    print(f"   Sequential: {sequential_time:.2f}s ({sequential_time/len(test_files):.2f}s per file)")
    print(f"   Parallel: {parallel_time:.2f}s ({parallel_time/len(test_files):.2f}s per file)")
    print(f"   Speedup: {speedup:.2f}x")
    print()

    # Cleanup
    for file in test_files:
        os.remove(file)
    os.rmdir(os.path.dirname(test_files[0]))


def demonstrate_progress_tracking():
    """Demonstrate detailed progress tracking."""
    print("=" * 60)
    print("DETAILED PROGRESS TRACKING")
    print("=" * 60)

    # Create a test file
    test_files = create_test_audio_files(1, 12.0)
    test_file = test_files[0]

    print(f"Analyzing with detailed progress: {os.path.basename(test_file)}")
    print()

    analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def detailed_progress_callback(progress: float, message: str = ""):
        # Create a simple progress bar
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r   [{bar}] {progress:6.1f}% - {message}", end="", flush=True)

    start_time = time.time()
    results = analyzer.analyze_file(
        test_file, comprehensive=True, progress_callback=detailed_progress_callback, detailed_progress=True
    )
    analysis_time = time.time() - start_time

    print()  # New line after progress bar
    print(f"   Completed in {analysis_time:.2f} seconds")
    print()

    # Show comprehensive results
    basic_info = results['basic_info']
    print("🎵 Analysis Results:")
    print(f"   Duration: {basic_info['duration']:.1f}s")
    print(f"   BPM: {basic_info['bpm']:.1f} (confidence: {basic_info['bpm_confidence']:.1f}%)")
    print(f"   Key: {basic_info['key']} (confidence: {basic_info['key_confidence']:.1f}%)")

    if 'chord_progression' in results:
        chords = results['chord_progression']
        print(f"   Main Progression: {' → '.join(chords.get('main_progression', [])[:4])}")

    if 'structure' in results:
        structure = results['structure']
        print(f"   Structure: {structure.get('form', 'Unknown')} ({structure.get('section_count', 0)} sections)")

    print()

    # Cleanup
    for file in test_files:
        os.remove(file)
    os.rmdir(os.path.dirname(test_file))


def main():
    """Run all demonstrations."""
    print("🎵 BPM Detector - Parallel Processing Demonstration")
    print()

    try:
        # System information
        demonstrate_system_info()

        # Single file analysis
        demonstrate_single_file_analysis()

        # Multiple file analysis
        demonstrate_multiple_file_analysis()

        # Progress tracking
        demonstrate_progress_tracking()

        print("=" * 60)
        print("DEMONSTRATION COMPLETED")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("✅ Automatic parallel configuration based on system capabilities")
        print("✅ Smart worker count adjustment for different workloads")
        print("✅ Detailed progress tracking for long-running analyses")
        print("✅ Performance improvements through parallel processing")
        print("✅ Graceful fallback to sequential processing when needed")
        print()
        print("Try running with different system loads to see adaptive behavior!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

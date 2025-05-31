#!/usr/bin/env python3
"""
Optimized BPM Detector Performance Demo

This script demonstrates the performance of the optimized BPM detector with enhanced settings.
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bpm_detector.parallel_analyzer import SmartParallelAudioAnalyzer
from bpm_detector.music_analyzer import AudioAnalyzer


def benchmark_analysis(audio_file: str, iterations: int = 3):
    """Run analysis benchmark"""

    print(f"🎵 Audio file: {audio_file}")
    print(f"🔄 Iterations: {iterations}")
    print("=" * 60)

    # Standard analyzer
    print("\n📊 Standard Analyzer (Sequential Processing)")
    standard_analyzer = AudioAnalyzer()
    standard_times = []

    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...", end=" ")
        start_time = time.time()

        try:
            standard_analyzer.analyze_file(
                audio_file, comprehensive=True, detect_key=True
            )
            elapsed = time.time() - start_time
            standard_times.append(elapsed)
            print(f"✅ {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ Error: {e}")
            return

    # Optimized parallel analyzer
    print("\n🚀 Optimized Parallel Analyzer")
    parallel_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)
    parallel_times = []

    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...", end=" ")
        start_time = time.time()

        try:
            parallel_analyzer.analyze_file(
                audio_file,
                comprehensive=True,
                progress_callback=None,  # Disable progress callback for accurate benchmarking
            )
            elapsed = time.time() - start_time
            parallel_times.append(elapsed)
            print(f"✅ {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ Error: {e}")
            return

    # Compare results
    print("\n" + "=" * 60)
    print("📈 Performance Comparison Results")
    print("=" * 60)

    avg_standard = sum(standard_times) / len(standard_times)
    avg_parallel = sum(parallel_times) / len(parallel_times)
    speedup = avg_standard / avg_parallel

    print("Standard Analyzer:")
    print(f"  Average time: {avg_standard:.2f}s")
    print(f"  Fastest time: {min(standard_times):.2f}s")
    print(f"  Slowest time: {max(standard_times):.2f}s")

    print("\nOptimized Parallel Analyzer:")
    print(f"  Average time: {avg_parallel:.2f}s")
    print(f"  Fastest time: {min(parallel_times):.2f}s")
    print(f"  Slowest time: {max(parallel_times):.2f}s")

    print(f"\n🎯 Speedup Factor: {speedup:.2f}x")

    if speedup > 1:
        improvement = (speedup - 1) * 100
        print(f"💡 {improvement:.1f}% performance improvement achieved!")
    else:
        print("⚠️  Parallel processing may have limited benefits in this environment")

    # Performance details
    performance_summary = parallel_analyzer.get_performance_summary()
    if performance_summary:
        print("\n📊 Detailed Performance Information:")
        for task, stats in performance_summary.items():
            print(f"  {task}:")
            print(f"    Average execution time: {stats['avg_execution_time']:.3f}s")
            print(f"    Execution count: {stats['execution_count']}")

    # Cleanup
    parallel_analyzer.cleanup()


def demo_multiple_files():
    """Demo parallel processing of multiple files"""

    print("\n" + "=" * 60)
    print("🎼 Multiple Files Parallel Processing Demo")
    print("=" * 60)

    # Find test files
    test_files = []
    examples_dir = Path(__file__).parent

    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        test_files.extend(examples_dir.glob(ext))

    if len(test_files) < 2:
        print("⚠️  Multiple test files not found")
        print("   Please place multiple audio files in the examples/ directory")
        return

    # Test up to 3 files
    test_files = [str(f) for f in test_files[:3]]

    print(f"📁 Number of test files: {len(test_files)}")
    for i, file in enumerate(test_files, 1):
        print(f"  {i}. {Path(file).name}")

    analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def progress_callback(progress, message):
        print(f"\r  Progress: {progress:3d}% - {message}", end="", flush=True)

    print("\n🚀 Starting parallel analysis...")
    start_time = time.time()

    try:
        results = analyzer.analyze_file(
            test_files, comprehensive=True, progress_callback=progress_callback
        )

        elapsed = time.time() - start_time
        print(f"\n✅ Complete! Total processing time: {elapsed:.2f}s")
        print(f"📊 Average per file: {elapsed/len(test_files):.2f}s")

        # Results summary
        print("\n📋 Analysis Results Summary:")
        for file_path, result in results.items():
            filename = Path(file_path).name
            if 'error' in result:
                print(f"  ❌ {filename}: Error - {result['error']}")
            else:
                basic_info = result.get('basic_info', {})
                bpm = basic_info.get('bpm', 'N/A')
                key = basic_info.get('key', 'N/A')
                print(f"  ✅ {filename}: BPM={bpm:.1f}, Key={key}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        analyzer.cleanup()


def main():
    """Main execution function"""

    print("🎵 BPM Detector Optimized Performance Demo")
    print("=" * 60)

    # Find test files
    examples_dir = Path(__file__).parent
    test_files = []

    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        test_files.extend(examples_dir.glob(ext))

    if not test_files:
        print("❌ No test files found")
        print("   Please place audio files in the examples/ directory")
        print("   Supported formats: MP3, WAV, FLAC, M4A")
        return

    # Single file benchmark
    test_file = str(test_files[0])
    benchmark_analysis(test_file)

    # Multiple files demo if available
    if len(test_files) > 1:
        demo_multiple_files()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("=" * 60)
    print("\n💡 Optimization Features Applied:")
    print("  • Faster audio loading with kaiser_fast resampling")
    print("  • Optimized BPM detection with audio length limiting")
    print("  • Enhanced key detection with efficient correlation")
    print("  • Aggressive parallel processing configuration")
    print("  • Memory-efficient processing with early cleanup")
    print("  • Feature caching for repeated computations")
    print("  • Dynamic system load monitoring")


if __name__ == "__main__":
    main()

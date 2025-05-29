"""Performance comparison between different analysis modes."""

import os
import sys
import numpy as np
import soundfile as sf
import time

# Add the src directory to the path so we can import bpm_detector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector import AudioAnalyzer


def create_test_audio(duration):
    """Create test audio of specified duration."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple chord progression
    chord_freqs = [
        [261.63, 329.63, 392.00],  # C major
        [220.00, 261.63, 329.63],  # A minor
        [174.61, 220.00, 261.63],  # F major
        [196.00, 246.94, 293.66]   # G major
    ]
    
    signal = np.zeros_like(t)
    chord_duration = duration / 4
    
    for i, chord in enumerate(chord_freqs):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        
        chord_mask = (t >= start_time) & (t < end_time)
        chord_t = t[chord_mask]
        
        chord_signal = np.zeros_like(chord_t)
        for freq in chord:
            chord_signal += 0.3 * np.sin(2 * np.pi * freq * chord_t)
        
        # Add rhythm
        beat_freq = 2.0
        beat_pattern = (np.sin(2 * np.pi * beat_freq * chord_t) > 0).astype(float)
        chord_signal *= (0.5 + 0.5 * beat_pattern)
        
        signal[chord_mask] = chord_signal
    
    # Add noise
    signal += 0.05 * np.random.randn(len(signal))
    signal = signal / np.max(np.abs(signal))
    
    return signal, sr


def benchmark_analysis():
    """Benchmark different analysis modes."""
    print("=== Performance Comparison ===\n")
    
    durations = [5, 10, 20, 30]  # Different audio lengths
    analyzer = AudioAnalyzer()
    
    results = []
    
    for duration in durations:
        print(f"Testing {duration}-second audio...")
        
        # Create test audio
        audio, sr = create_test_audio(duration)
        test_file = f"test_{duration}s.wav"
        sf.write(test_file, audio, sr)
        
        try:
            # Basic analysis
            start_time = time.time()
            basic_results = analyzer.analyze_file(test_file, comprehensive=False)
            basic_time = time.time() - start_time
            
            # Comprehensive analysis
            start_time = time.time()
            comp_results = analyzer.analyze_file(test_file, comprehensive=True)
            comp_time = time.time() - start_time
            
            results.append({
                'duration': duration,
                'basic_time': basic_time,
                'comprehensive_time': comp_time,
                'ratio': comp_time / basic_time
            })
            
            print(f"  Basic: {basic_time:.2f}s, Comprehensive: {comp_time:.2f}s, Ratio: {comp_time/basic_time:.1f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    print("\n=== Summary ===")
    print("Duration | Basic Time | Comprehensive Time | Speed Ratio")
    print("-" * 55)
    for r in results:
        print(f"{r['duration']:8}s | {r['basic_time']:10.2f}s | {r['comprehensive_time']:18.2f}s | {r['ratio']:10.1f}x")
    
    # Performance recommendations
    print("\n=== Recommendations ===")
    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"Average speed difference: {avg_ratio:.1f}x")
    
    if avg_ratio > 5:
        print("⚠️  Comprehensive analysis is significantly slower")
        print("💡 Consider using basic analysis for real-time applications")
    elif avg_ratio > 3:
        print("⚡ Moderate performance impact")
        print("💡 Acceptable for batch processing")
    else:
        print("✅ Good performance balance")
    
    print("\n🚀 Performance Tips:")
    print("1. Use comprehensive=False for BPM/Key only (fastest)")
    print("2. Shorter audio files process much faster")
    print("3. Consider chunking long files for progress updates")
    print("4. Cache results for repeated analysis")


if __name__ == "__main__":
    benchmark_analysis()
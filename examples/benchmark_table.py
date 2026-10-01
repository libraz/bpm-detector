"""Reproduce the performance table published in the README.

The README quotes basic, sequential-comprehensive and parallel-comprehensive
times per audio length. Those numbers belong to the machine that ran them, so
they are measured here rather than copied forward. Run this and paste the table
it prints, together with the machine line above it.

    python examples/benchmark_table.py
"""

import os
import platform
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpm_detector import AudioAnalyzer, SmartParallelAudioAnalyzer  # noqa: E402
from performance_comparison import create_test_audio  # noqa: E402

DURATIONS = [5, 10, 20, 30]


def timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def warmup(sequential, parallel):
    """Pay the numba/librosa JIT cost before anything is timed.

    Without this the first measured row absorbs several seconds of compilation
    and reads as slower than a longer clip, which is how the published table
    ended up non-monotonic.
    """
    audio, sr = create_test_audio(2)
    path = "bench_warmup.wav"
    sf.write(path, audio, sr)
    try:
        sequential.analyze_file(path, comprehensive=False)
        sequential.analyze_file(path, comprehensive=True)
        parallel.analyze_file(path, comprehensive=True)
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    sequential = AudioAnalyzer()
    parallel = SmartParallelAudioAnalyzer(auto_parallel=True)
    warmup(sequential, parallel)

    print(f"{platform.platform()}, Python {platform.python_version()}, "
          f"{os.cpu_count()} logical cores")
    print()
    print("| Audio Length | Basic Analysis | Sequential Comprehensive | "
          "Parallel Comprehensive | Parallel Speedup |")
    print("|--------------|----------------|--------------------------|"
          "------------------------|------------------|")

    for duration in DURATIONS:
        audio, sr = create_test_audio(duration)
        test_file = f"bench_{duration}s.wav"
        sf.write(test_file, audio, sr)
        try:
            basic = timed(lambda: sequential.analyze_file(test_file, comprehensive=False))
            seq = timed(lambda: sequential.analyze_file(test_file, comprehensive=True))
            par = timed(lambda: parallel.analyze_file(test_file, comprehensive=True))
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

        print(f"| {duration} seconds | {basic:.1f}s | {seq:.1f}s | {par:.1f}s | "
              f"{seq / par:.1f}x |")


if __name__ == "__main__":
    main()

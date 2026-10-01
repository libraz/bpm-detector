# BPM & Key Detector

[![CI](https://img.shields.io/github/actions/workflow/status/libraz/bpm-detector/ci.yml?branch=main&label=CI)](https://github.com/libraz/bpm-detector/actions)
[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/libraz/bpm-detector/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python tool for automatic music analysis including BPM, key detection, and advanced music production features.

## Related: libsonare

The same author maintains **[libsonare](https://github.com/libraz/libsonare)** — a dependency-free audio DSP toolkit (C++ / Python / WASM, Apache-2.0) that ports the algorithms here to C++ and adds broadcast-grade mastering, mixing, and editing DSP. It is being developed into a headless DAW.

🎵 **[Try the music analysis demo](https://libsonare.libraz.net/music-analysis.html)** — BPM, key, chord, and structure analysis running entirely in your browser. No install required. ([all demos](https://libsonare.libraz.net/demos.html))

This project remains available as a stable, lightweight, MIT-licensed Python-only option for BPM/key detection. New development is focused on libsonare.

## Features

### Core Analysis
- **BPM Detection**: High-precision tempo detection algorithm
  - Automatic fast/slow layer selection
  - Harmonic clustering
  - Confidence scoring
- **Key Detection**: Music theory-based key detection
  - Uses Krumhansl-Schmuckler key profiles
  - Supports both major and minor keys
  - Chroma feature-based analysis
  - When confidence is low, `key` is `None` and the tentative candidate (e.g. `Unknown(E♭?)`) is kept in `key_detection_details`

### Advanced Music Analysis
- **Chord Progression Analysis**: Automatic chord detection and harmonic analysis
  - Chord sequence identification (C-Am-F-G)
  - Functional harmony analysis (I-vi-IV-V)
  - Modulation detection
  - Chord complexity scoring
  - Detects major, minor, 7th, and sus4 chords on all 12 roots
- **Song Structure Analysis**: Automatic section detection and form analysis
  - Section boundaries (intro, verse, chorus, bridge)
  - Song form identification (ABABCB)
  - Repetition pattern detection
  - Structural complexity analysis
- **Rhythm & Groove Analysis**: Detailed rhythmic pattern analysis
  - Time signature detection (4/4, 3/4, 6/8, etc.)
  - Groove type classification (straight, swing, shuffle)
  - Syncopation level measurement
  - Rhythmic complexity scoring
- **Timbre & Instrumentation**: Audio texture and instrument analysis
  - Instrument classification (piano, guitar, drums, etc.)
  - Timbral characteristics (brightness, warmth, roughness)
  - Effects usage detection (reverb, distortion, chorus)
  - Acoustic density analysis
- **Melody & Harmony Analysis**: Musical content analysis
  - Melodic range and contour analysis
  - Harmonic complexity measurement
  - Consonance/dissonance evaluation
  - Interval distribution analysis
- **Dynamics & Energy**: Audio dynamics and energy profiling
  - Dynamic range analysis
  - Energy profile generation
  - Climax point detection
  - Loudness analysis
- **Music Production Reference**: Automated reference sheet generation
  - Production notes and recommendations
  - Similar track characteristics
  - Reference tags for music commissioning
  - Feature vector generation for similarity matching

### Advanced Structure Analysis
- **Section Classification**: Intelligent musical section detection
  - Automatic identification of intro, verse, chorus, bridge, outro
  - Context-aware classification using audio characteristics
  - Vocal presence detection and spoken word identification
  - Energy building detection for dynamic sections
- **Boundary Detection**: Precise structural boundary identification
  - Self-similarity matrix analysis for section boundaries
  - Beat-aligned boundary snapping for musical accuracy
  - Repetition pattern detection and analysis
  - Novelty-based boundary detection algorithm
- **Section Processing**: Advanced post-processing and refinement
  - Smart section merging based on duration and characteristics
  - Spectral analysis for instrumental subtype classification
  - Form analysis with letter notation (ABABCB)
  - Structural complexity scoring

### Parallel Processing & Performance
- **Smart Parallel Processing**: Automatic CPU-based optimization
  - Auto-detection of system capabilities (CPU cores, memory, load)
  - Adaptive worker count based on system performance
  - Dynamic load monitoring and adjustment
  - Graceful fallback to sequential processing
- **Progress Tracking**: Real-time progress monitoring
  - Hierarchical progress display for parallel tasks
  - Detailed progress for each analysis module
  - Time estimation and performance metrics
  - Interactive progress bars with task status
- **Performance Optimization**: Comprehensive analysis spread across cores
  - Measured figures and how to reproduce them: [Performance Benchmarks](#performance-benchmarks)
  - Intelligent memory management
  - Process vs thread pool selection based on workload

### Selective Analysis
- **Flexible Analysis Options**: Analyze only what you need
  - `--rhythm`: Analyze rhythm and time signature only
  - `--chords`: Analyze chord progressions only
  - `--structure`: Analyze musical structure only
  - `--timbre`: Analyze timbre and instruments only
  - `--melody`: Analyze melody and harmony only
  - `--dynamics`: Analyze dynamics only
  - Mix and match options for custom analysis pipelines
- **Performance Benefits**: Faster analysis by skipping unnecessary computations
  - Significantly reduced processing time for targeted analysis
  - Lower memory usage
  - Ideal for batch processing when only specific features are needed

## Quick Links

- 📦 [PyPI Package](https://pypi.org/project/bpm-detector/) (Coming Soon)
- 🐳 [Docker Image](https://github.com/libraz/bpm-detector/pkgs/container/bpm-detector)
- 📊 [Test Coverage](https://codecov.io/gh/libraz/bpm-detector)
- 🔧 [CI/CD Status](https://github.com/libraz/bpm-detector/actions)
- 📖 [Documentation](https://github.com/libraz/bpm-detector)
- 🐛 [Issues](https://github.com/libraz/bpm-detector/issues)
- 💡 [Feature Requests](https://github.com/libraz/bpm-detector/issues/new)

## Installation

### Option 1: Install from PyPI (Coming Soon)

```bash
pip install bpm-detector
```

### Option 2: Install from Source (Current)

```bash
# Clone the repository
git clone git@github.com:libraz/bpm-detector.git
cd bpm-detector

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Option 3: Using rye (if you have rye installed)

```bash
# Clone the repository
git clone git@github.com:libraz/bpm-detector.git
cd bpm-detector

# Install dependencies with rye
rye sync
```

## Usage

### Command Line Interface

After installation, you can use the `bpm-detector` command:

```bash
# Basic usage (BPM only)
bpm-detector your_audio_file.wav

# With key detection
bpm-detector --detect-key your_audio_file.wav

# Multiple files
bpm-detector --detect-key *.wav *.mp3

# Suppress progress output (-q)
bpm-detector --quiet --detect-key your_audio_file.wav

# Selective analysis - Analyze only what you need for faster results
bpm-detector --rhythm your_audio_file.wav  # BPM + time signature only
bpm-detector --detect-key --rhythm your_audio_file.wav  # BPM + key + rhythm
bpm-detector --melody --timbre your_audio_file.wav  # BPM + melody + instruments
bpm-detector --rhythm --chords --structure your_audio_file.wav  # Multiple analyses

# Available selective analysis options:
#   --rhythm    : Analyze rhythm and time signature
#   --chords    : Analyze chord progressions
#   --structure : Analyze musical structure
#   --timbre    : Analyze timbre and instruments
#   --melody    : Analyze melody and harmony
#   --dynamics  : Analyze dynamics

# Comprehensive analysis (all analyses; key detection still needs --detect-key)
bpm-detector --comprehensive your_audio_file.wav

# Parallel processing
bpm-detector --comprehensive your_audio_file.wav  # Auto-parallel enabled by default
bpm-detector --comprehensive --max-workers 4 your_audio_file.wav  # Manual worker count
bpm-detector --comprehensive --no-parallel your_audio_file.wav  # Disable parallel processing
bpm-detector --show-system-info  # Show system capabilities and parallel configuration
bpm-detector --comprehensive --detailed-progress your_audio_file.wav  # Detailed progress tracking
```

### Python API

#### Basic Analysis (Fast)
```python
from bpm_detector import AudioAnalyzer

# Initialize analyzer
analyzer = AudioAnalyzer()

# Basic analysis (BPM + Key only) - Fast!
results = analyzer.analyze_file('song.wav', detect_key=True, comprehensive=False)

print(f"BPM: {results['basic_info']['bpm']:.1f}")
print(f"Key: {results['basic_info']['key']}")
print(f"Duration: {results['basic_info']['duration']:.1f} seconds")
```

#### Selective Analysis (Fast & Efficient)
```python
from bpm_detector import AudioAnalyzer

analyzer = AudioAnalyzer()

# Analyze only rhythm and time signature (fastest)
results = analyzer.analyze_file(
    'song.wav',
    detect_key=False,
    comprehensive=False,
    analyze_rhythm=True
)
print(f"BPM: {results['basic_info']['bpm']:.1f}")
print(f"Time Signature: {results['rhythm']['time_signature']}")
print(f"Groove: {results['rhythm']['groove_type']}")

# Multiple selective analyses
results = analyzer.analyze_file(
    'song.wav',
    detect_key=True,
    comprehensive=False,
    analyze_rhythm=True,
    analyze_melody=True,
    analyze_timbre=True
)

# Access selected analysis results
print(f"Key: {results['basic_info']['key']}")
print(f"Time: {results['rhythm']['time_signature']}")
print(f"Instruments: {results['timbre']['dominant_instruments']}")
print(f"Vocal Range: {results['melody_harmony']['melodic_range']}")

# Available selective analysis parameters:
#   analyze_rhythm=True    : Rhythm and time signature
#   analyze_chords=True    : Chord progressions
#   analyze_structure=True : Musical structure
#   analyze_timbre=True    : Timbre and instruments
#   analyze_melody=True    : Melody and harmony
#   analyze_dynamics=True  : Dynamics
```

#### Comprehensive Analysis (Detailed)
```python
# Comprehensive analysis - All features!
results = analyzer.analyze_file('song.wav', comprehensive=True)

# Basic info
basic = results['basic_info']
print(f"BPM: {basic['bpm']:.1f}, Key: {basic['key']}")

# Chord progression
chords = results['chord_progression']
print(f"Main progression: {' → '.join(chords['main_progression'])}")
print(f"Chord complexity: {chords['chord_complexity']:.1%}")

# Song structure
structure = results['structure']
print(f"Form: {structure['form']}")
print(f"Sections: {structure['section_count']}")

# Rhythm analysis
rhythm = results['rhythm']
print(f"Time signature: {rhythm['time_signature']}")
print(f"Groove: {rhythm['groove_type']}")

# Generate production reference sheet
reference_sheet = analyzer.generate_reference_sheet(results)
print(reference_sheet)
```

#### Smart Parallel Processing
```python
from bpm_detector import SmartParallelAudioAnalyzer

# Smart parallel analyzer with auto-optimization
analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

# Single file with progress tracking
def progress_callback(progress, message):
    print(f"Progress: {progress:.1f}% - {message}")

results = analyzer.analyze_file(
    'song.wav',
    comprehensive=True,
    progress_callback=progress_callback
)

# Multiple files with parallel processing (returns {path: results})
files = ['song1.wav', 'song2.wav', 'song3.wav']
batch_results = analyzer.analyze_files(files, comprehensive=True)

# Manual configuration
analyzer = SmartParallelAudioAnalyzer(
    auto_parallel=True,
    max_workers=4  # Override automatic worker count
)

# Check system configuration
from bpm_detector import AutoParallelConfig
config = AutoParallelConfig.get_optimal_config()
print(f"Parallel enabled: {config.enable_parallel}")
print(f"Max workers: {config.max_workers}")
print(f"Strategy: {config.strategy.value}")
```

#### Performance Comparison
```python
import time
from bpm_detector import AudioAnalyzer, SmartParallelAudioAnalyzer

# Traditional analyzer
traditional = AudioAnalyzer()
parallel = SmartParallelAudioAnalyzer(auto_parallel=True)

# Basic analysis
start = time.time()
basic_results = traditional.analyze_file('song.wav', comprehensive=False)
print(f"Basic analysis: {time.time() - start:.2f}s")

# Sequential comprehensive analysis
start = time.time()
sequential_results = traditional.analyze_file('song.wav', comprehensive=True)
sequential_time = time.time() - start
print(f"Sequential comprehensive: {sequential_time:.2f}s")

# Parallel comprehensive analysis
start = time.time()
parallel_results = parallel.analyze_file('song.wav', comprehensive=True)
parallel_time = time.time() - start
print(f"Parallel comprehensive: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

Measured timings are in [Performance Benchmarks](#performance-benchmarks).

### Docker Usage

You can also run the detector using Docker:

```bash
# Pull the latest image
docker pull ghcr.io/libraz/bpm-detector:latest

# Run with audio files (mount your audio directory)
docker run --rm -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest --detect-key audio.wav

# Show CLI help (the default command)
docker run --rm ghcr.io/libraz/bpm-detector:latest
```

### Development Mode

If you're running from source without installation:

```bash
# Using Python module
python -m bpm_detector.cli your_audio_file.wav

# Using rye
rye run python -m bpm_detector.cli your_audio_file.wav

# Build Docker image locally
docker build -t bpm-detector .
docker run --rm -v $(pwd):/workspace bpm-detector --help
```

## Options

### Command Line Options

#### Basic Options
- `--detect-key`: Enable key detection
- `--detect-modulation`: Detect key changes over time and print each modulation (requires `--detect-key`)
- `--comprehensive`: Enable comprehensive music analysis
- `--quiet`, `-q`: Suppress progress output
- `--sr SR`: Sample rate (default: 22050)
- `--hop HOP`: Hop length (default: 128)
- `--min_bpm MIN_BPM`: Minimum BPM (default: 40.0)
- `--max_bpm MAX_BPM`: Maximum BPM (default: 300.0)
- `--start_bpm START_BPM`: Starting BPM (default: 150.0)

#### Selective Analysis Options
- `--rhythm`, `--chords`, `--structure`, `--timbre`, `--melody`, `--dynamics`: Run only the selected analyses (combinable; `--comprehensive` enables all of them)

#### Parallel Processing Options
- `--auto-parallel`: Enable automatic parallel optimization (default: enabled)
- `--no-parallel`: Disable parallel processing
- `--max-workers N`: Override automatic worker count
- `--detailed-progress`: Show detailed progress for each analysis task
- `--show-system-info`: Show system information and parallel configuration

### Python API Options
```python
analyzer.analyze_file(
    path='song.wav',
    detect_key=True,           # Enable key detection (default: True)
    detect_modulation=False,   # Key changes over time in results['modulation_analysis'] (requires detect_key)
    comprehensive=True,        # Enable all advanced features (default: True)
    min_bpm=40.0,              # Minimum BPM range
    max_bpm=300.0,             # Maximum BPM range
    start_bpm=150.0,           # Starting BPM estimate
    progress_callback=None,    # Progress callback function
    analyze_rhythm=False,      # Selective analyses (used when comprehensive=False)
    analyze_chords=False,
    analyze_structure=False,
    analyze_timbre=False,
    analyze_melody=False,
    analyze_dynamics=False,
)
```

`detect_key` and `comprehensive` default to `True` in the API, unlike the CLI where both are opt-in. Pass `comprehensive=False` for a fast BPM/key-only run.

## Output Examples

### Basic Analysis Output
```
song.mp3
  > Estimated BPM : 132.50 BPM  (conf 98.4%)
  > Estimated Key : F# Major  (conf 62.4%)
```

### Comprehensive Analysis Output
```
song.mp3
  > Duration: 262.1s, BPM: 132.5, Key: F# Major
  > Chord Progression: F# → C# → F# → D#m
  > Structure: IARBARBARBCAC (13 sections)
  > Section Details (13 sections):
    1. Intro (00:00, 3bars, mid E, mid C)
    2. Verse(A-melo) (00:05, 15bars, mid E, mid C)
    3. Pre_Chorus(B-melo) (00:32, 15bars, mid E, mid C)
    4. Chorus(Sabi) (00:59, 9bars, mid E, mid C)
    ...
  > Rhythm: 4/4 time, straight groove
  > Instruments: vocals, piano, guitar
  > Timbre: Brightness 0.6, Warmth 1.0
  > Melody: 64.9% coverage, 4.4 octave range
  > Full Range: D#2 - G#6 (Tenor)
  > Vocal Range: C#3 - G#5 (Tenor)
  > Harmony: 77.3% consonance, 68.4% complexity
  > Dynamics: 80.0dB range, 44.5% variation

  > Estimated BPM : 132.50 BPM  (conf 98.4%)
  > Estimated Key : F# Major  (conf 62.4%)
```

### Reference Sheet Example (excerpt)
```markdown
# Music Production Reference Sheet

## Basic Information
- **Tempo**: 120.0 BPM
- **Key**: C Major
- **Time Signature**: 4/4
- **Duration**: 180 seconds

## Song Structure
- **Section Count**: 7
- **Structural Complexity**: 0.6
- **Repetition Ratio**: 57.1%

## Harmony & Chord Progression
- **Main Chord Progression**: C - Am - F - G
- **Chord Complexity**: 65.0%
- **Harmonic Rhythm**: 2.0 changes/sec

...

## Production Notes
- Arrangement Density: medium
- Production Style: rock_pop
- Mix Characteristics: bright_mix, punchy_drums

## Reference Tags
upbeat, major-key, piano-driven, guitar-driven, mid-energy
```

Note: The actual CLI output includes colors:
- File names are displayed in bright cyan
- Analysis summary lines are colored by category (blue, magenta, cyan, yellow)
- Final estimates in bright green (BPM) and magenta (Key)

## Performance & Technical Details

### Performance Benchmarks

#### Sequential vs Parallel Processing

Measured on macOS 26.6 (arm64, 18 logical cores) with Python 3.12.9, on the
synthetic chord-progression clips the harness generates. Times are a single run
after a warmup pass, so they exclude librosa's first-call JIT compilation.

| Audio Length | Basic Analysis | Sequential Comprehensive | Parallel Comprehensive | Parallel Speedup |
|--------------|----------------|-------------------------|------------------------|------------------|
| 5 seconds    | 0.3s          | 3.3s                   | 2.6s                   | 1.3x             |
| 10 seconds   | 0.7s          | 6.8s                   | 5.1s                   | 1.3x             |
| 20 seconds   | 1.3s          | 13.3s                  | 10.5s                  | 1.3x             |
| 30 seconds   | 1.9s          | 20.0s                  | 15.6s                  | 1.3x             |

Analysis time scales with clip length, and the parallel speedup is bounded by
the modules that cannot be split rather than by core count — adding cores past
a handful buys little. Reproduce the table on your own machine with
`python examples/benchmark_table.py`.

#### Auto-Parallel Configuration
`AutoParallelConfig` picks a strategy from the logical core count and the current load:

| Logical Cores | Strategy | Workers | Pool |
|---------------|----------|---------|------|
| 8+            | Aggressive parallel | cores − 2 | Process |
| 4-7           | Balanced parallel (conservative with 2 workers under high load) | up to 8 | Thread |
| 2-3           | Conservative parallel (sequential under load or low memory) | up to 3 | Thread |
| 1             | Sequential only | 1 | — |

Run `bpm-detector --show-system-info` to see what it selects on your machine.

**Recommendations**:
- Use `comprehensive=False` for real-time applications
- Use `SmartParallelAudioAnalyzer` for batch processing and detailed analysis
- Let auto-parallel optimization handle system-specific tuning

### BPM Detection Algorithm
- Uses librosa's tempo detection functionality
- Harmonic clustering for candidate integration
- Automatic selection of higher layers (×1.5, ×2)

### Key Detection Algorithm
- Chroma features extraction
- Correlation calculation with Krumhansl-Schmuckler key profiles
- Optimal selection from 24 keys (12 major + 12 minor)

### Advanced Analysis Algorithms
- **Chord Detection**: Template matching with chroma features and harmonic clustering
- **Structure Analysis**: Self-similarity matrix with novelty-based boundary detection
- **Section Classification**: Context-aware classification using energy, spectral, and vocal features
- **Boundary Detection**: Beat-aligned boundary snapping with repetition analysis
- **Section Processing**: Smart merging and spectral-based refinement
- **Rhythm Analysis**: Onset detection with time signature and groove classification
- **Timbre Analysis**: MFCC, spectral contrast, and instrument classification
- **Melody Analysis**: Fundamental frequency tracking with pitch stability analysis
- **Harmony Analysis**: Consonance/dissonance evaluation and harmonic complexity
- **Dynamics Analysis**: RMS energy profiling with climax detection
- **Similarity Engine**: Multi-dimensional feature vector generation and comparison
- **Parallel Processing**: Adaptive CPU-based optimization with dynamic load balancing

### Why Comprehensive Analysis is Optional
Comprehensive analysis is implemented as an optional feature for several reasons:

1. **Performance**: Advanced analysis adds significant computational overhead — roughly 10x the basic run in the table above
2. **Use Cases**: Many users only need BPM/key for DJ mixing, tempo matching, or basic analysis
3. **Processing Time**: For batch processing, users can choose faster basic analysis when detailed features aren't needed
4. **Flexibility**: Allows users to balance between speed and feature completeness based on requirements

## Project Stats

- **Tests**: 400+ tests across every analysis module
- **Supported Python**: 3.12+ (CI runs 3.12 and 3.14)
- **Supported Formats**: WAV, MP3, FLAC, OGG via libsndfile; M4A and other formats need FFmpeg installed (included in the Docker image)

## Dependencies

Requires Python 3.12 or later. Runtime dependencies and their minimum versions are declared in [`pyproject.toml`](pyproject.toml), and the tested set is pinned in `requirements.lock`.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

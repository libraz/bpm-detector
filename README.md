# BPM & Key Detector

[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python tool for automatic music analysis including BPM, key detection, and advanced music production features.

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

### Advanced Music Analysis (NEW!)
- **Chord Progression Analysis**: Automatic chord detection and harmonic analysis
  - Chord sequence identification (C-Am-F-G)
  - Functional harmony analysis (I-vi-IV-V)
  - Modulation detection
  - Chord complexity scoring
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

## Quick Links

- 📦 [PyPI Package](https://pypi.org/project/bpm-detector/) (Coming Soon)
- 🐳 [Docker Image](https://github.com/libraz/bpm-detector/pkgs/container/bpm-detector)
- 📊 [Test Coverage](https://codecov.io/gh/libraz/bpm-detector)
- 🔧 [CI/CD Status](https://github.com/libraz/bpm-detector/actions)
- 📖 [Documentation](https://github.com/libraz/bpm-detector)
- 🐛 [Issues](https://github.com/libraz/bpm-detector/issues)
- 💡 [Feature Requests](https://github.com/libraz/bpm-detector/issues/new?template=feature_request.md)

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

# Show progress bar
bpm-detector --progress --detect-key your_audio_file.wav
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

#### Performance Comparison
```python
import time

# Fast analysis (0.1-0.7 seconds)
start = time.time()
basic_results = analyzer.analyze_file('song.wav', comprehensive=False)
print(f"Basic analysis: {time.time() - start:.2f}s")

# Comprehensive analysis (2.5-15 seconds depending on audio length)
start = time.time()
full_results = analyzer.analyze_file('song.wav', comprehensive=True)
print(f"Comprehensive analysis: {time.time() - start:.2f}s")
```

### Docker Usage

You can also run the detector using Docker:

```bash
# Pull the latest image
docker pull ghcr.io/libraz/bpm-detector:latest

# Run with audio files (mount your audio directory)
docker run --rm -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest --detect-key audio.wav

# Interactive mode
docker run --rm -it -v /path/to/your/audio:/workspace ghcr.io/libraz/bpm-detector:latest
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
- `--detect-key`: Enable key detection
- `--comprehensive`: Enable comprehensive music analysis (NEW!)
- `--progress`: Show progress bar
- `--sr SR`: Sample rate (default: 22050)
- `--hop HOP`: Hop length (default: 128)
- `--min_bpm MIN_BPM`: Minimum BPM (default: 40.0)
- `--max_bpm MAX_BPM`: Maximum BPM (default: 300.0)
- `--start_bpm START_BPM`: Starting BPM (default: 150.0)

### Python API Options
```python
analyzer.analyze_file(
    path='song.wav',
    detect_key=True,           # Enable key detection
    comprehensive=True,        # Enable all advanced features
    min_bpm=40.0,             # Minimum BPM range
    max_bpm=300.0,            # Maximum BPM range
    start_bpm=150.0,          # Starting BPM estimate
    progress_callback=None     # Progress callback function
)
```

## Output Examples

### Basic Analysis Output
```
example.wav
  > BPM Candidates Top10
  * 120.00 BPM : 45
    240.00 BPM : 23
     60.00 BPM : 18
    180.00 BPM : 12
    ...
  > Estimated BPM : 120.00 BPM  (conf 78.3%)
  > Estimated Key : C Major  (conf 85.2%)
```

### Comprehensive Analysis Output
```
example.wav
  > BPM: 120.0, Key: C Major, Duration: 180.0s
  > Chord Progression: C → Am → F → G (I-vi-IV-V)
  > Structure: Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus (ABABCB)
  > Rhythm: 4/4 time, straight groove, moderate syncopation
  > Instruments: Piano-driven, guitar, drums
  > Energy: Mid-level, climax at 2:30
```

### Reference Sheet Example
```markdown
# Music Production Reference Sheet

## Basic Information
- **Tempo**: 120.0 BPM
- **Key**: C Major
- **Time Signature**: 4/4
- **Duration**: 180 seconds

## Harmony & Chord Progression
- **Main Chord Progression**: C - Am - F - G
- **Chord Complexity**: 65.0%
- **Harmonic Rhythm**: 2.0 changes/sec

## Production Notes
- Arrangement Density: medium
- Production Style: rock_pop
- Mix Characteristics: bright_mix, punchy_drums

## Reference Tags
upbeat, major-key, piano-driven, guitar-driven, mid-energy
```

Note: The actual CLI output includes colors:
- File names are displayed in bright cyan
- Section headers ("> BPM Candidates") in yellow
- Selected BPM candidates (*) in green
- Final estimates in bright green (BPM) and magenta (Key)

## Performance & Technical Details

### Performance Benchmarks
| Audio Length | Basic Analysis | Comprehensive Analysis | Speed Ratio |
|--------------|----------------|------------------------|-------------|
| 5 seconds    | 0.7s          | 2.5s                  | 3.4x        |
| 10 seconds   | 0.1s          | 4.9s                  | 47x         |
| 20 seconds   | 0.2s          | 9.9s                  | 43x         |
| 30 seconds   | 0.3s          | 15.0s                 | 45x         |

**Recommendation**: Use `comprehensive=False` for real-time applications, `comprehensive=True` for detailed analysis.

### BPM Detection Algorithm
- Uses librosa's tempo detection functionality
- Harmonic clustering for candidate integration
- Automatic selection of higher layers (×1.5, ×2)

### Key Detection Algorithm
- Chroma features extraction
- Correlation calculation with Krumhansl-Schmuckler key profiles
- Optimal selection from 24 keys (12 major + 12 minor)

### Advanced Analysis Algorithms
- **Chord Detection**: Template matching with chroma features
- **Structure Analysis**: Self-similarity matrix with boundary detection
- **Rhythm Analysis**: Onset detection with pattern recognition
- **Timbre Analysis**: MFCC and spectral feature extraction
- **Melody Analysis**: Fundamental frequency tracking with librosa.pyin
- **Dynamics Analysis**: RMS energy and spectral energy profiling

### Why Comprehensive Analysis is Optional
Comprehensive analysis is implemented as an optional feature for several reasons:

1. **Performance**: Advanced analysis adds significant computational overhead (3-45x slower)
2. **Use Cases**: Many users only need BPM/key for DJ mixing, tempo matching, or basic analysis
3. **Processing Time**: For batch processing, users can choose faster basic analysis when detailed features aren't needed
4. **Flexibility**: Allows users to balance between speed and feature completeness based on requirements

## Project Stats

- **Test Coverage**: 100% (54/54 tests passing)
- **Supported Python**: 3.12+
- **Docker Image Size**: ~1.6GB
- **Build Time**: ~4 minutes
- **Supported Formats**: WAV, MP3, FLAC, M4A, OGG
- **Analysis Features**: 7 comprehensive modules + similarity engine

## Dependencies

### Core Dependencies
- librosa >= 0.11.0
- soundfile >= 0.13.1
- numpy >= 2.2.6
- tqdm >= 4.67.1
- audioread >= 3.0.1
- colorama >= 0.4.6

### Advanced Analysis Dependencies
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pandas >= 2.0.0

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

# BPM & Key Detector

[![codecov](https://codecov.io/gh/libraz/bpm-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/libraz/bpm-detector)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for automatic detection of BPM (tempo) and musical key from audio files.

## Features

- **BPM Detection**: High-precision tempo detection algorithm
  - Automatic fast/slow layer selection
  - Harmonic clustering
  - Confidence scoring
- **Key Detection**: Music theory-based key detection
  - Uses Krumhansl-Schmuckler key profiles
  - Supports both major and minor keys
  - Chroma feature-based analysis

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

You can also use the detector programmatically:

```python
from bpm_detector import AudioAnalyzer

# Initialize analyzer
analyzer = AudioAnalyzer()

# Analyze a file
results = analyzer.analyze_file('your_audio_file.wav', detect_key=True)

print(f"BPM: {results['bpm']:.2f}")
print(f"Key: {results['key']}")
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

- `--detect-key`: Enable key detection
- `--progress`: Show progress bar
- `--sr SR`: Sample rate (default: 22050)
- `--hop HOP`: Hop length (default: 128)
- `--min_bpm MIN_BPM`: Minimum BPM (default: 40.0)
- `--max_bpm MAX_BPM`: Maximum BPM (default: 300.0)
- `--start_bpm START_BPM`: Starting BPM (default: 150.0)

## Output Example

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

Note: The actual output includes colors:
- File names are displayed in bright cyan
- Section headers ("> BPM Candidates") in yellow
- Selected BPM candidates (*) in green
- Final estimates in bright green (BPM) and magenta (Key)

## Technical Details

### BPM Detection Algorithm
- Uses librosa's tempo detection functionality
- Harmonic clustering for candidate integration
- Automatic selection of higher layers (×1.5, ×2)

### Key Detection Algorithm
- Chroma features extraction
- Correlation calculation with Krumhansl-Schmuckler key profiles
- Optimal selection from 24 keys (12 major + 12 minor)

### Why Key Detection is Optional
Key detection is implemented as an optional feature (`--detect-key`) for several reasons:

1. **Performance**: Key detection adds computational overhead due to chroma feature extraction and correlation calculations
2. **Use Cases**: Many users only need BPM detection for DJ mixing, tempo matching, or rhythm analysis
3. **Processing Time**: For batch processing of many files, users can choose faster BPM-only analysis when key information isn't needed
4. **Flexibility**: Allows users to balance between speed and feature completeness based on their specific requirements

## Project Stats

- **Test Coverage**: 90%+ (171 lines covered)
- **Supported Python**: 3.12
- **Docker Image Size**: ~1.6GB
- **Build Time**: ~4 minutes
- **Supported Formats**: WAV, MP3, FLAC, M4A, OGG

## Dependencies

- librosa >= 0.11.0
- soundfile >= 0.13.1
- numpy >= 2.2.6
- tqdm >= 4.67.1
- audioread >= 3.0.1
- colorama >= 0.4.6

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

# Contributing to BPM & Key Detector

Thank you for your interest in contributing to BPM & Key Detector! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Basic understanding of audio processing and music theory (helpful but not required)

### Areas for Contribution

We welcome contributions in the following areas:

1. **Core Analysis Features**
   - Improving BPM detection accuracy
   - Enhancing key detection algorithms
   - Adding new chord progression analysis features
   - Optimizing performance

2. **New Analysis Modules**
   - Additional rhythm pattern detection
   - More sophisticated instrument classification
   - Advanced harmonic analysis
   - Genre classification

3. **Performance Optimization**
   - Reducing processing time
   - Memory usage optimization
   - Parallel processing implementation

4. **Documentation**
   - API documentation improvements
   - Usage examples
   - Tutorial content
   - Translation to other languages

5. **Testing**
   - Adding test cases
   - Improving test coverage
   - Performance benchmarks

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone git@github.com:libraz/bpm-detector.git
cd bpm-detector
```

### 2. Set Up Development Environment

#### Option A: Using rye (Recommended)

```bash
# Install dependencies with rye
rye sync

# Activate the virtual environment
rye shell
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy
```

### 3. Verify Installation

```bash
# Run basic tests
python -m pytest tests/

# Test CLI functionality
python -m bpm_detector.cli examples/test_audio.wav --detect-key
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Development Guidelines

#### For Core Analysis Features

- Maintain backward compatibility
- Add comprehensive tests for new features
- Document any new parameters or options
- Consider performance implications

#### For New Analysis Modules

- Follow the existing module structure (see `src/bpm_detector/`)
- Implement the standard `analyze()` method
- Add appropriate error handling
- Include docstrings with parameter descriptions

#### Example Module Structure

```python
"""New analysis module."""

import numpy as np
from typing import Dict, Any


class NewAnalyzer:
    """New analysis functionality."""
    
    def __init__(self, hop_length: int = 128):
        """Initialize analyzer.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length
    
    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Analysis results dictionary
        """
        # Implementation here
        return {}
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src/bpm_detector

# Run specific test file
python -m pytest tests/test_chord_analyzer.py

# Run with verbose output
python -m pytest -v
```

### Writing Tests

- Add tests for all new functionality
- Use descriptive test names
- Include edge cases and error conditions
- Test with various audio formats and lengths

#### Example Test Structure

```python
import unittest
import numpy as np
from src.bpm_detector.new_analyzer import NewAnalyzer


class TestNewAnalyzer(unittest.TestCase):
    """Test cases for NewAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = NewAnalyzer()
        self.sr = 22050
        self.test_signal = np.random.randn(self.sr * 5)  # 5 seconds
    
    def test_basic_functionality(self):
        """Test basic analysis functionality."""
        result = self.analyzer.analyze(self.test_signal, self.sr)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('expected_key', result)
        
    def test_empty_input(self):
        """Test behavior with empty input."""
        empty_signal = np.array([])
        result = self.analyzer.analyze(empty_signal, self.sr)
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
```

### Performance Testing

```bash
# Run performance benchmarks
python examples/performance_comparison.py

# Profile specific functions
python -m cProfile -s cumulative your_script.py
```

## Submitting Changes

### 1. Before Submitting

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Commit messages are clear

### 2. Commit Guidelines

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add chord progression complexity scoring"
git commit -m "Fix memory leak in structure analyzer"
git commit -m "Optimize similarity matrix computation"

# Include issue numbers when applicable
git commit -m "Fix BPM detection for short files (fixes #123)"
```

### 3. Pull Request Process

1. Push your branch to your fork
2. Create a pull request against the main repository
3. Fill out the pull request template
4. Wait for review and address feedback
5. Ensure CI checks pass

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Other (please describe)

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Tested with various audio files

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (justified)

## Documentation
- [ ] Updated relevant documentation
- [ ] Added docstrings for new functions
- [ ] Updated README if needed
```

## Code Style

### Python Style Guidelines

We follow PEP 8 with some modifications:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Key Style Points

- Line length: 88 characters (black default)
- Use type hints for function parameters and return values
- Write descriptive docstrings for all public functions
- Use meaningful variable names
- Add comments for complex algorithms

### Example Code Style

```python
def analyze_chord_progression(
    chroma: np.ndarray, 
    sr: int, 
    hop_length: int = 128
) -> Dict[str, Any]:
    """Analyze chord progression from chroma features.
    
    Args:
        chroma: Chroma feature matrix (12 x n_frames)
        sr: Sample rate in Hz
        hop_length: Hop length for frame timing
        
    Returns:
        Dictionary containing:
            - main_progression: List of chord names
            - complexity: Complexity score (0-1)
            - confidence: Detection confidence (0-1)
            
    Raises:
        ValueError: If chroma matrix has wrong dimensions
    """
    if chroma.shape[0] != 12:
        raise ValueError(f"Expected 12 chroma bins, got {chroma.shape[0]}")
    
    # Implementation here
    return {
        'main_progression': [],
        'complexity': 0.0,
        'confidence': 0.0
    }
```

## Documentation

### API Documentation

- Use Google-style docstrings
- Include parameter types and descriptions
- Document return values and exceptions
- Provide usage examples

### README Updates

When adding new features:

1. Update feature list in README.md
2. Add usage examples
3. Update performance benchmarks if applicable
4. Update README_ja.md (Japanese version)

### Code Comments

- Explain complex algorithms
- Document non-obvious design decisions
- Include references to papers or algorithms used

## Reporting Issues

### Bug Reports

Include the following information:

- Python version
- Operating system
- Audio file format and characteristics
- Complete error message and stack trace
- Minimal code example to reproduce

### Feature Requests

- Describe the use case
- Explain the expected behavior
- Consider implementation complexity
- Discuss potential performance impact

### Performance Issues

- Include benchmark results
- Specify audio file characteristics
- Compare with expected performance
- Suggest potential optimizations

## Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Docker image builds successfully

## Getting Help

- **Questions**: Open a GitHub issue with the "question" label
- **Discussions**: Use GitHub Discussions for general topics
- **Real-time chat**: Join our Discord server (link in README)

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation credits

Thank you for contributing to BPM & Key Detector! 🎵
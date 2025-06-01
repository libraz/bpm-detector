"""Timbre and instrument analysis module."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .effects_detector import EffectsDetector
from .instrument_classifier import InstrumentClassifier
from .timbre_features import TimbreFeatureExtractor


class TimbreAnalyzer:
    """Analyzes timbral features and instrument presence."""

    # Spectral feature ranges for classification
    FEATURE_RANGES = {
        'brightness': {'low': 0.3, 'high': 0.7},
        'roughness': {'low': 0.2, 'high': 0.8},
        'warmth': {'low': 0.3, 'high': 0.7},
    }

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize timbre analyzer.

        Args:
            hop_length: Hop length for analysis
            n_fft: FFT size
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Initialize component analyzers
        self.feature_extractor = TimbreFeatureExtractor(hop_length, n_fft)
        self.instrument_classifier = InstrumentClassifier(hop_length, n_fft)
        self.effects_detector = EffectsDetector(hop_length)

    def extract_timbral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive timbral features.

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Dictionary of timbral features
        """
        return self.feature_extractor.extract_timbral_features(y, sr)

    def analyze_brightness(self, spectral_centroid: np.ndarray, sr: int) -> float:
        """Analyze spectral brightness.

        Args:
            spectral_centroid: Spectral centroid values
            sr: Sample rate

        Returns:
            Brightness score (0-1)
        """
        return self.feature_extractor.analyze_brightness(spectral_centroid, sr)

    def analyze_roughness(self, spectral_contrast: np.ndarray) -> float:
        """Analyze spectral roughness.

        Args:
            spectral_contrast: Spectral contrast values

        Returns:
            Roughness score (0-1)
        """
        return self.feature_extractor.analyze_roughness(spectral_contrast)

    def analyze_warmth(self, mfcc: np.ndarray) -> float:
        """Analyze timbral warmth.

        Args:
            mfcc: MFCC features

        Returns:
            Warmth score (0-1)
        """
        return self.feature_extractor.analyze_warmth(mfcc)

    def analyze_density(self, features: Dict[str, np.ndarray]) -> float:
        """Analyze acoustic density/texture.

        Args:
            features: Dictionary of timbral features

        Returns:
            Density score (0-1)
        """
        return self.feature_extractor.analyze_density(features)

    def classify_instruments(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Classify instruments present in the audio.

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            List of detected instruments with confidence scores
        """
        return self.instrument_classifier.classify_instruments(y, sr)

    def analyze_effects_usage(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze usage of audio effects.

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Dictionary of effect usage scores
        """
        return self.effects_detector.analyze_effects_usage(y, sr)

    def analyze_texture(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze overall acoustic texture.

        Args:
            features: Dictionary of timbral features

        Returns:
            Dictionary of texture characteristics
        """
        return self.feature_extractor.analyze_texture(features)

    def analyze(self, y: np.ndarray, sr: int, progress_callback=None) -> Dict[str, Any]:
        """Perform complete timbre analysis.

        Args:
            y: Audio signal
            sr: Sample rate
            progress_callback: Optional progress callback function

        Returns:
            Complete timbre analysis results
        """
        if progress_callback:
            progress_callback(10, "Starting timbre analysis...")

        # Extract timbral features
        if progress_callback:
            progress_callback(20, "Extracting spectral features...")
        features = self.extract_timbral_features(y, sr)

        if progress_callback:
            progress_callback(40, "Computing brightness and warmth...")

        # Analyze basic timbral characteristics
        brightness = self.analyze_brightness(features['spectral_centroid'], sr)
        warmth = self.analyze_warmth(features['mfcc'])

        if progress_callback:
            progress_callback(55, "Computing roughness and density...")

        roughness = self.analyze_roughness(features['spectral_contrast'])
        density = self.analyze_density(features)

        if progress_callback:
            progress_callback(70, "Detecting instruments...")

        # Classify instruments
        instruments = self.classify_instruments(y, sr)

        if progress_callback:
            progress_callback(85, "Analyzing audio effects...")

        # Analyze effects usage
        effects = self.analyze_effects_usage(y, sr)

        if progress_callback:
            progress_callback(95, "Computing texture analysis...")

        # Analyze texture
        texture = self.analyze_texture(features)

        if progress_callback:
            progress_callback(100, "Timbre analysis complete")

        return {
            'brightness': brightness,
            'roughness': roughness,
            'warmth': warmth,
            'density': density,
            'dominant_instruments': instruments[:5],  # Top 5 instruments
            'effects_usage': effects,
            'texture': texture,
            'spectral_features': {
                'centroid_mean': float(np.mean(features['spectral_centroid'])),
                'bandwidth_mean': float(np.mean(features['spectral_bandwidth'])),
                'rolloff_mean': float(np.mean(features['spectral_rolloff'])),
                'flatness_mean': float(np.mean(features['spectral_flatness'])),
                'zcr_mean': float(np.mean(features['zero_crossing_rate'])),
            },
        }

    # Backward compatibility methods for existing tests
    def _filter_redundant_instruments(self, instruments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out redundant instrument detections (backward compatibility)."""
        return self.instrument_classifier._filter_redundant_instruments(instruments)

    def _calculate_instrument_confidence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        instrument: Optional[str] = None,
        harmonic: Optional[np.ndarray] = None,
        percussive: Optional[np.ndarray] = None,
        spectral_shape=None,
    ) -> float:
        """Calculate confidence for instrument presence (backward compatibility)."""
        return self.instrument_classifier._calculate_instrument_confidence(
            magnitude, freqs, freq_range, low_freq, high_freq, instrument, harmonic, percussive, spectral_shape
        )

    def _calculate_instrument_prominence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> float:
        """Calculate instrument prominence in the mix (backward compatibility)."""
        return self.instrument_classifier._calculate_instrument_prominence(
            magnitude, freqs, freq_range, low_freq, high_freq
        )

    def _detect_reverb(self, y: np.ndarray, sr: int) -> float:
        """Detect reverb presence (backward compatibility)."""
        return self.effects_detector._detect_reverb(y, sr)

    def _detect_distortion(self, y: np.ndarray, sr: int) -> float:
        """Detect distortion presence (backward compatibility)."""
        return self.effects_detector._detect_distortion(y, sr)

    def _detect_chorus(self, y: np.ndarray, sr: int) -> float:
        """Detect chorus/modulation effects (backward compatibility)."""
        return self.effects_detector._detect_chorus(y, sr)

    def _detect_compression(self, y: np.ndarray) -> float:
        """Detect compression presence (backward compatibility)."""
        return self.effects_detector._detect_compression(y)

    # Add INSTRUMENT_RANGES for backward compatibility
    @property
    def INSTRUMENT_RANGES(self):
        """Get instrument frequency ranges (backward compatibility)."""
        return self.instrument_classifier.INSTRUMENT_RANGES

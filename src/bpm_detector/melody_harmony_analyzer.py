"""Melody and harmony analysis module - Main coordinator."""

import numpy as np
from typing import Dict, Any, Optional

from .music_theory import (
    PROGRESS_KEY_DETECTION, PROGRESS_MELODY_EXTRACTION, 
    PROGRESS_MELODIC_ANALYSIS, PROGRESS_HARMONIC_ANALYSIS, 
    PROGRESS_HARMONIC_RHYTHM
)
from .melody_analyzer import MelodyAnalyzer
from .harmony_analyzer import HarmonyAnalyzer
from .key_detector import KeyDetector


class MelodyHarmonyAnalyzer:
    """Main coordinator for melody and harmony analysis."""
    
    def __init__(self, hop_length: int = 512, fmin: float = 80.0, fmax: float = 2000.0,
                 consonance_ratings: Optional[Dict[int, float]] = None):
        """Initialize melody harmony analyzer.
        
        Args:
            hop_length: Hop length for analysis
            fmin: Minimum frequency for pitch tracking
            fmax: Maximum frequency for pitch tracking
            consonance_ratings: Optional custom consonance ratings for intervals
        """
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        
        # Set default consonance ratings if none provided
        if consonance_ratings is None:
            consonance_ratings = {0: 1.0, 7: 0.8, 4: 0.7, 3: 0.6, 8: 0.5, 9: 0.4}
        self.consonance_ratings = consonance_ratings  # Store for test compatibility
        
        # Initialize component analyzers
        self.melody_analyzer = MelodyAnalyzer(hop_length, fmin, fmax)
        self.harmony_analyzer = HarmonyAnalyzer(hop_length, consonance_ratings)
        self.key_detector = KeyDetector(hop_length)
    
    def analyze(
        self,
        y: np.ndarray,
        sr: int,
        *,
        key_hint: Optional[str] = None,
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform complete melody and harmony analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            key_hint: Optional external key hint for validation
            progress_callback: Optional progress callback function
            **kwargs: Additional parameters
            
        Returns:
            Complete melody and harmony analysis results
        """
        # Progress tracking with fine-grained steps
        if progress_callback:
            progress_callback(PROGRESS_KEY_DETECTION, "Detecting key...")
        
        # Detect key (use external hint if available)
        key_detection = self.key_detector.detect_key(y, sr, external_key_hint=key_hint)
        
        if progress_callback:
            progress_callback(PROGRESS_MELODY_EXTRACTION, "Extracting melody...")
        
        # Extract melody
        melody = self.melody_analyzer.extract_melody(y, sr)
        
        if progress_callback:
            progress_callback(PROGRESS_MELODIC_ANALYSIS, "Analyzing melodic characteristics...")
        
        # Analyze melodic characteristics
        melodic_range = self.melody_analyzer.analyze_melodic_range(melody)
        melodic_direction = self.melody_analyzer.analyze_melodic_direction(melody)
        interval_distribution = self.melody_analyzer.analyze_interval_distribution(melody)
        pitch_stability = self.melody_analyzer.analyze_pitch_stability(melody)
        
        if progress_callback:
            progress_callback(PROGRESS_HARMONIC_ANALYSIS, "Analyzing harmonic characteristics...")
        
        # Analyze harmonic characteristics
        harmony_complexity = self.harmony_analyzer.analyze_harmony_complexity(y, sr)
        consonance = self.harmony_analyzer.analyze_consonance(y, sr)
        
        if progress_callback:
            progress_callback(PROGRESS_HARMONIC_RHYTHM, "Analyzing harmonic rhythm...")
        
        harmonic_rhythm = self.harmony_analyzer.analyze_harmonic_rhythm(y, sr)
        
        # Compile results with organized structure expected by tests
        melody_section = {
            'range': melodic_range,
            'direction': melodic_direction,
            'intervals': interval_distribution,
            'stability': pitch_stability,
            'melody_present': bool(np.any(melody['voiced_flag'])),
            'melody_coverage': float(np.mean(melody['voiced_flag'])) if len(melody['voiced_flag']) > 0 else 0.0
        }
        
        harmony_section = {
            'complexity': harmony_complexity,
            'consonance': consonance,
            'rhythm': harmonic_rhythm
        }
        
        # Calculate combined features
        combined_features = {
            'melody_harmony_balance': self._calculate_melody_harmony_balance(melody_section, harmony_section),
            'overall_complexity': self._calculate_overall_complexity(melody_section, harmony_section),
            'musical_sophistication': self._calculate_musical_sophistication(melody_section, harmony_section, key_detection)
        }
        
        result = {
            'key_detection': key_detection,
            'melody': melody_section,
            'harmony': harmony_section,
            'combined_features': combined_features,
            # Keep backward compatibility
            'melodic_range': melodic_range,
            'melodic_direction': melodic_direction,
            'interval_distribution': interval_distribution,
            'pitch_stability': pitch_stability,
            'harmony_complexity': harmony_complexity,
            'consonance': consonance,
            'harmonic_rhythm': harmonic_rhythm,
            'melody_present': bool(np.any(melody['voiced_flag'])),
            'melody_coverage': float(np.mean(melody['voiced_flag'])) if len(melody['voiced_flag']) > 0 else 0.0
        }

        # Optional: return raw melody information
        if kwargs.get("return_raw_melody"):
            result["raw_melody"] = melody

        return result
    
    # Convenience methods for direct access to component functionality
    def extract_melody(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract melody line from audio."""
        return self.melody_analyzer.extract_melody(y, sr)
    
    def detect_key(self, y: np.ndarray, sr: int, *, external_key_hint: Optional[str] = None) -> Dict[str, Any]:
        """Detect musical key."""
        return self.key_detector.detect_key(y, sr, external_key_hint=external_key_hint)
    
    def analyze_harmony_complexity(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic complexity."""
        return self.harmony_analyzer.analyze_harmony_complexity(y, sr)
    
    def analyze_consonance(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic consonance/dissonance."""
        return self.harmony_analyzer.analyze_consonance(y, sr)
    
    def analyze_harmonic_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze harmonic rhythm."""
        return self.harmony_analyzer.analyze_harmonic_rhythm(y, sr)
    
    def _calculate_melody_harmony_balance(self, melody_section: Dict[str, Any], harmony_section: Dict[str, Any]) -> float:
        """Calculate balance between melody and harmony complexity."""
        try:
            melody_complexity = melody_section.get('direction', {}).get('contour_complexity', 0.0)
            harmony_complexity = harmony_section.get('complexity', {}).get('harmonic_complexity', 0.0)
            
            # Balance score: closer to 0.5 means better balance
            if melody_complexity + harmony_complexity == 0:
                return 0.5
            
            balance = min(melody_complexity, harmony_complexity) / max(melody_complexity, harmony_complexity)
            return float(balance)
        except (KeyError, ZeroDivisionError, TypeError):
            return 0.5
    
    def _calculate_overall_complexity(self, melody_section: Dict[str, Any], harmony_section: Dict[str, Any]) -> float:
        """Calculate overall musical complexity."""
        try:
            melody_complexity = melody_section.get('direction', {}).get('contour_complexity', 0.0)
            harmony_complexity = harmony_section.get('complexity', {}).get('harmonic_complexity', 0.0)
            interval_complexity = len([k for k, v in melody_section.get('intervals', {}).items()
                                     if v > 0 and k not in ['small_intervals', 'medium_intervals', 'large_intervals']]) / 12.0
            
            # Weighted combination
            overall = (melody_complexity * 0.4 + harmony_complexity * 0.4 + interval_complexity * 0.2)
            return float(min(1.0, overall))
        except (KeyError, TypeError):
            return 0.0
    
    def _calculate_musical_sophistication(self, melody_section: Dict[str, Any], harmony_section: Dict[str, Any], key_detection: Dict[str, Any]) -> float:
        """Calculate overall musical sophistication."""
        try:
            # Check for musical content quality first
            melody_coverage = melody_section.get('melody_coverage', 0.0)
            key_confidence = key_detection.get('confidence', 0.0) / 100.0
            
            # If very low musical content, return low sophistication
            if melody_coverage < 0.3 or key_confidence < 0.3:
                return min(0.3, melody_coverage * key_confidence)
            
            # Melody sophistication factors
            melody_range = melody_section.get('range', {}).get('range_octaves', 0.0)
            melody_complexity = melody_section.get('direction', {}).get('contour_complexity', 0.0)
            interval_variety = len([k for k, v in melody_section.get('intervals', {}).items()
                                  if v > 0 and k not in ['small_intervals', 'medium_intervals', 'large_intervals']]) / 12.0
            
            # Harmony sophistication factors
            harmony_complexity = harmony_section.get('complexity', {}).get('harmonic_complexity', 0.0)
            consonance_balance = abs(0.5 - harmony_section.get('consonance', {}).get('consonance_score', 0.5))  # Balance between consonance/dissonance
            harmonic_rhythm = min(1.0, harmony_section.get('rhythm', {}).get('harmonic_rhythm', 0.0) / 5.0)  # Normalize
            
            # Weighted combination with quality gates
            sophistication = (
                melody_range * 0.15 +
                melody_complexity * 0.2 +
                interval_variety * 0.15 +
                harmony_complexity * 0.2 +
                consonance_balance * 0.1 +
                harmonic_rhythm * 0.1 +
                key_confidence * 0.1
            )
            
            # Apply quality multiplier
            quality_multiplier = min(1.0, (melody_coverage + key_confidence) / 2.0)
            sophistication *= quality_multiplier
            
            return float(min(1.0, sophistication))
        except (KeyError, TypeError):
            return 0.0
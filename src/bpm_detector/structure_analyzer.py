"""Musical structure analysis module - Main coordinator."""

import numpy as np
from typing import List, Dict, Any

from .boundary_detector import BoundaryDetector
from .section_classifier import SectionClassifier
from .section_processor import SectionProcessor


class StructureAnalyzer:
    """Analyzes musical structure and form - Main coordinator."""

    # Enhanced section types with more detailed categories for J-Pop analysis
    SECTION_TYPES = {
        'intro': {'energy_range': (0.1, 0.4), 'complexity_range': (0.1, 0.5)},
        'verse': {'energy_range': (0.3, 0.7), 'complexity_range': (0.4, 0.8)},
        'pre_chorus': {'energy_range': (0.5, 0.8), 'complexity_range': (0.5, 0.8)},
        'chorus': {'energy_range': (0.6, 1.0), 'complexity_range': (0.5, 0.9)},
        'bridge': {'energy_range': (0.4, 0.8), 'complexity_range': (0.6, 1.0)},
        'outro': {'energy_range': (0.1, 0.5), 'complexity_range': (0.2, 0.6)},
        'instrumental': {'energy_range': (0.3, 0.9), 'complexity_range': (0.3, 0.8)},
        'break': {'energy_range': (0.2, 0.5), 'complexity_range': (0.2, 0.5)},
        'interlude': {'energy_range': (0.4, 0.7), 'complexity_range': (0.5, 0.8)},
        'solo': {'energy_range': (0.5, 0.9), 'complexity_range': (0.6, 1.0)},
        'spoken': {'energy_range': (0.1, 0.4), 'complexity_range': (0.1, 0.4)},
    }

    # ASCII label definitions (J-Pop terminology)
    JP_ASCII_LABELS = {
        "intro": "Intro",
        "verse": "A-melo",
        "pre_chorus": "B-melo",
        "chorus": "Sabi",
        "bridge": "C-melo",
        "instrumental": "Kansou",
        "break": "Break",
        "interlude": "Interlude",
        "solo": "Solo",
        "spoken": "Serifu",
        "outro": "Outro",
    }

    def __init__(self, hop_length: int = 512, frame_size: int = 4096):
        """Initialize structure analyzer.

        Args:
            hop_length: Hop length for analysis
            frame_size: Frame size for feature extraction
        """
        self.hop_length = hop_length
        self.frame_size = frame_size

        # Initialize component analyzers
        self.boundary_detector = BoundaryDetector(hop_length)
        self.section_classifier = SectionClassifier(hop_length)
        self.section_processor = SectionProcessor(hop_length)

    def analyze(self, y: np.ndarray, sr: int, bpm: float = 130.0) -> Dict[str, Any]:
        """Perform complete structural analysis.

        Args:
            y: Audio signal
            sr: Sample rate
            bpm: BPM for dynamic segment length calculation

        Returns:
            Complete structural analysis results
        """
        # Calculate adaptive energy scale for this track
        self.section_processor.calculate_energy_scale(y)
        # Note: _energy_scale attribute is no longer used in the new modular design

        # Extract features
        features = self.boundary_detector.extract_structural_features(y, sr)

        # Compute self-similarity matrix
        similarity_matrix = self.boundary_detector.compute_self_similarity_matrix(
            features
        )

        # Detect boundaries with dynamic segment length
        boundaries = self.boundary_detector.detect_boundaries(
            similarity_matrix, sr, bpm=bpm
        )

        # Beat snap alignment
        boundaries = self.boundary_detector.snap_to_beat(boundaries, sr, bpm)

        # Classify sections with similarity matrix for verse repetition detection
        sections = self.section_classifier.classify_sections(
            y, sr, boundaries, similarity_matrix
        )

        # Merge and denoise with enhanced processing including fade detection
        sections = self.section_processor.post_process_sections(
            sections, bpm=bpm, y=y, sr=sr
        )

        # Refine section labels using spectral analysis
        sections = self.section_processor.refine_section_labels_with_spectral_analysis(
            y, sr, sections
        )

        # Analyze form
        form_analysis = self.section_processor.analyze_form(sections)

        # Detect repetitions
        repetitions = self.boundary_detector.detect_repetitions(similarity_matrix, sr)

        # Generate summary text
        summary_txt = self.section_processor.summarize_sections(sections)

        return {
            'sections': sections,
            'section_summary': summary_txt,
            'form': form_analysis.get('form', ''),
            'repetition_ratio': form_analysis.get('repetition_ratio', 0.0),
            'structural_complexity': form_analysis.get('structure_complexity', 0.0),
            'section_count': form_analysis.get('section_count', len(sections)),
            'unique_sections': form_analysis.get('unique_sections', 0),
            'repetitions': repetitions,
            'boundaries': [b * self.hop_length / sr for b in boundaries],
        }

    # Convenience methods for direct access to component functionality
    def extract_structural_features(
        self, y: np.ndarray, sr: int
    ) -> Dict[str, np.ndarray]:
        """Extract features for structural analysis."""
        return self.boundary_detector.extract_structural_features(y, sr)

    def compute_self_similarity_matrix(
        self, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute self-similarity matrix from features."""
        return self.boundary_detector.compute_self_similarity_matrix(features)

    def detect_boundaries(
        self,
        similarity_matrix: np.ndarray,
        sr: int,
        min_segment_length: float = 12.0,
        bpm: float = 130.0,
    ) -> List[int]:
        """Detect structural boundaries using novelty detection."""
        return self.boundary_detector.detect_boundaries(
            similarity_matrix, sr, min_segment_length, bpm
        )

    def classify_sections(
        self, y: np.ndarray, sr: int, boundaries: List[int], bpm: float = 130.0
    ) -> List[Dict[str, Any]]:
        """Classify sections based on their characteristics."""
        return self.section_classifier.classify_sections(y, sr, boundaries)

    def analyze_form(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall musical form."""
        return self.section_processor.analyze_form(sections)

    def detect_repetitions(
        self, similarity_matrix: np.ndarray, sr: int
    ) -> List[Dict[str, Any]]:
        """Detect repeated sections in the music."""
        return self.boundary_detector.detect_repetitions(similarity_matrix, sr)

    def summarize_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Generate a summary text of sections for display."""
        return self.section_processor.summarize_sections(sections)

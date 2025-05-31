"""Section post-processing module for musical structure analysis."""

import numpy as np
from typing import List, Dict, Any

from .jpop_structure_optimizer import JPopStructureOptimizer
from .section_analyzer import SectionAnalyzer


class SectionProcessor:
    """Post-processes and refines musical sections."""

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

    # Instrumental aliases (Solo, Interlude, etc.) to be normalized
    _INSTRUMENTAL_ALIASES = {"solo", "interlude", "buildup", "breakdown"}

    def __init__(self, hop_length: int = 512):
        """Initialize section processor.

        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length
        self.jpop_optimizer = JPopStructureOptimizer()
        self.analyzer = SectionAnalyzer(hop_length)

    def post_process_sections(
        self,
        raw: List[Dict[str, Any]],
        total_duration: float = None,
        merge_threshold: float = None,
        bpm: float = 130.0,
        y: np.ndarray = None,
        sr: int = None,
    ) -> List[Dict[str, Any]]:
        """Enhanced post-process sections with fade detection and outro refinement.

        Args:
            raw: List of raw sections
            total_duration: Total duration of the audio (optional)
            merge_threshold: Minimum duration threshold for merging (optional)
            bpm: BPM for duration calculations
            y: Audio signal for fade detection (optional)
            sr: Sample rate for fade detection (optional)

        Returns:
            List of processed sections
        """
        # Handle backward compatibility - if total_duration is actually bpm (old signature)
        if (
            isinstance(total_duration, (int, float))
            and merge_threshold is None
            and len(raw) > 0
        ):
            # Old signature: post_process_sections(sections, bpm)
            bpm = total_duration
            total_duration = None

        # Add duration field to sections if missing
        for section in raw:
            if 'duration' not in section:
                section['duration'] = section['end_time'] - section['start_time']

        # Use merge_threshold if provided, otherwise calculate based on BPM
        if merge_threshold is not None:
            min_dur = merge_threshold
            very_short_thresh = merge_threshold / 2.0
        else:
            # Calculate 4-bar duration as minimum section length (stricter)
            four_bar_duration = (16 * 60.0) / bpm  # 16 beats = 4 bars
            min_dur = max(
                6.0, four_bar_duration
            )  # Full 4 bars, minimum 6 seconds (stricter)

            # Also calculate 2-bar threshold for very short segments
            two_bar_duration = (8 * 60.0) / bpm  # 8 beats = 2 bars
            very_short_thresh = max(3.0, two_bar_duration)

        # FIRST: Apply strict chorus chain limitation BEFORE any merging
        chorus_limited = []
        consecutive_chorus_count = 0
        for i, sec in enumerate(raw):
            if sec["type"] == "chorus":
                consecutive_chorus_count += 1
                if consecutive_chorus_count > 2:
                    # Convert excess chorus to instrumental
                    sec = sec.copy()  # Don't modify original
                    sec["type"] = "instrumental"
                    sec["ascii_label"] = self.JP_ASCII_LABELS["instrumental"]
                    sec["vocal_ratio"] = 0.2  # Mark as low vocal
                    consecutive_chorus_count = 0  # Reset counter
            else:
                consecutive_chorus_count = 0
            chorus_limited.append(sec)

        # SECOND: Break consecutive chorus chains and restore instrumentals
        if y is not None and sr is not None:
            enhanced = self.jpop_optimizer.break_consecutive_chorus_chains(
                chorus_limited, y, sr, bpm
            )
        else:
            enhanced = chorus_limited.copy()

        # SECOND: merge adjacent same types (but preserve intentional breaks and instrumental sections)
        merged = []
        for seg in enhanced:
            # Only merge if types are same AND there's no significant gap AND not restored instrumental
            should_merge = False
            if merged and seg['type'] == merged[-1]['type']:
                # Never merge restored instrumentals (those with vocal_ratio)
                if seg['type'] == 'instrumental' and seg.get('vocal_ratio') is not None:
                    should_merge = False
                elif (
                    merged[-1]['type'] == 'instrumental'
                    and merged[-1].get('vocal_ratio') is not None
                ):
                    should_merge = False
                else:
                    # Check for time gap (if gap > 1 second, don't merge)
                    time_gap = seg['start_time'] - merged[-1]['end_time']
                    if abs(time_gap) <= 1.0:  # Allow small gaps due to rounding
                        should_merge = True

            if should_merge:
                # Merge sections of same type
                merged[-1]['end_time'] = seg['end_time']
                merged[-1]['duration'] = (
                    merged[-1]['end_time'] - merged[-1]['start_time']
                )
                # Update ASCII label to match merged type
                merged[-1]['ascii_label'] = self.JP_ASCII_LABELS.get(
                    merged[-1]['type'], merged[-1]['type']
                )
            else:
                merged.append(seg)

        # === B-D-B merging will be executed later ===
        # merged = self._merge_chorus_instrumental_chorus(merged, bpm)

        # Enhanced pass: aggressive short segment merging
        enhanced = []
        for i, seg in enumerate(merged):
            # Very aggressive merging for segments < 2 bars
            if seg['duration'] < very_short_thresh:
                if len(enhanced) > 0:
                    # Always merge very short segments into previous
                    enhanced[-1]['end_time'] = seg['end_time']
                    enhanced[-1]['duration'] = (
                        enhanced[-1]['end_time'] - enhanced[-1]['start_time']
                    )
                    continue

            # Enhanced D absorption rule: Always absorb 8 bars after Chorus/Verse/Bridge
            eight_bar_duration = (8 * 4 * 60.0) / bpm  # 8 bars duration

            if seg['type'] == 'instrumental':
                # Protect restored instrumentals from absorption
                if seg.get('vocal_ratio') is not None:
                    enhanced.append(seg)
                    continue

                should_absorb = False

                # Check if this instrumental follows Chorus/Verse/Bridge
                if len(enhanced) > 0:
                    prev_type = enhanced[-1]['type']
                    if prev_type in ['chorus', 'verse', 'bridge']:
                        # Only absorb normal D sections without vocal_ratio
                        if seg['duration'] <= eight_bar_duration:
                            should_absorb = True
                        # Also absorb short instrumentals (original logic)
                        elif seg['duration'] < min_dur:
                            should_absorb = True

                if should_absorb:
                    enhanced[-1]['end_time'] = seg['end_time']
                    enhanced[-1]['duration'] = (
                        enhanced[-1]['end_time'] - enhanced[-1]['start_time']
                    )
                    continue

                # Forward merging for remaining short instrumentals
                if seg['duration'] < min_dur and i + 1 < len(merged):
                    next_seg = merged[i + 1]
                    if next_seg['type'] in ['chorus', 'verse']:
                        # Extend next section to include this instrumental
                        next_seg['start_time'] = seg['start_time']
                        next_seg['duration'] = (
                            next_seg['end_time'] - next_seg['start_time']
                        )
                        continue

            # Standard short segment absorption (4-bar minimum) - but protect restored instrumentals
            should_absorb = seg['duration'] < min_dur and len(enhanced) > 0

            # Protect restored instrumentals (those with vocal_ratio information)
            if seg['type'] == 'instrumental' and seg.get('vocal_ratio') is not None:
                should_absorb = False

            # Also protect if previous section would absorb an instrumental
            if (
                len(enhanced) > 0
                and enhanced[-1]['type'] == 'chorus'
                and seg['type'] == 'instrumental'
                and seg.get('vocal_ratio') is not None
            ):
                should_absorb = False

            if should_absorb:
                # Absorb short segment into previous section
                enhanced[-1]['end_time'] = seg['end_time']
                enhanced[-1]['duration'] = (
                    enhanced[-1]['end_time'] - enhanced[-1]['start_time']
                )

                # If absorbing changes the character significantly, update type
                if (
                    seg['duration'] > min_dur * 0.3
                ):  # If absorbed segment is substantial
                    # Keep the type of the longer segment
                    if seg['duration'] > enhanced[-1]['duration'] * 0.5:
                        # Absorbed segment is significant, consider hybrid classification
                        prev_type = enhanced[-1]['type']
                        curr_type = seg['type']

                        # Apply smart merging rules
                        merged_type = self._smart_merge_types(
                            prev_type,
                            curr_type,
                            enhanced[-1]['duration'],
                            seg['duration'],
                        )
                        enhanced[-1]['type'] = merged_type
                        enhanced[-1]['ascii_label'] = self.JP_ASCII_LABELS.get(
                            merged_type, merged_type
                        )
            else:
                enhanced.append(seg)

        # Enhanced outro detection with fade analysis
        if y is not None and sr is not None and len(enhanced) > 0:
            enhanced = self.analyzer.enhance_outro_detection(enhanced, y, sr)

        # Apply R→B pairing rules first
        enhanced = self.jpop_optimizer.enforce_pre_chorus_chorus_pairing(enhanced)

        # Apply Pre-Chorus consecutive suppression filter AFTER pairing (order changed)
        enhanced = self.jpop_optimizer.suppress_consecutive_pre_chorus(enhanced)

        # Apply short bridge downgrade (< 12 bars → verse)
        enhanced = self._downgrade_short_bridges(enhanced, bpm)

        # Apply ending instrumental cleanup
        enhanced = self._cleanup_ending_instrumentals(enhanced, bpm)

        # Apply chorus hook detection
        enhanced = self.analyzer.detect_chorus_hooks(enhanced)

        # Collapse A-R alternating patterns
        enhanced = self.jpop_optimizer.collapse_alternating_ar_patterns(enhanced)

        # === ⑥ oversized pre_chorus split ==============================
        MAX_R_BARS = 16  # 16 bars(≈32s) を超える R は強制 Verse+R
        max_r_sec = MAX_R_BARS * 4 * 60 / bpm
        fixed = []
        for sec in enhanced:
            if sec["type"] == "pre_chorus" and sec["duration"] > max_r_sec:
                # Calculate how many segments we need
                num_segments = int(np.ceil(sec["duration"] / max_r_sec))
                segment_duration = sec["duration"] / num_segments

                current_start = sec["start_time"]
                for i in range(num_segments):
                    segment = sec.copy()
                    segment["start_time"] = current_start
                    segment["end_time"] = current_start + segment_duration
                    segment["duration"] = segment_duration

                    # First segment becomes verse, others remain pre_chorus
                    if i == 0:
                        segment["type"] = "verse"
                        segment["ascii_label"] = self.JP_ASCII_LABELS["verse"]
                    else:
                        segment["type"] = "pre_chorus"
                        segment["ascii_label"] = self.JP_ASCII_LABELS["pre_chorus"]

                    fixed.append(segment)
                    current_start += segment_duration
            else:
                fixed.append(sec)
        enhanced = fixed

        # === ⑦ oversize A/R split (Verse & Pre-Chorus) ===========
        SPLIT_BARS = 16
        split_sec = SPLIT_BARS * 4 * 60 / bpm
        final = []
        for sec in enhanced:
            if sec["type"] in ["verse", "pre_chorus"] and sec["duration"] > split_sec:
                mid = sec["start_time"] + sec["duration"] / 2
                first = sec.copy()
                last = sec.copy()
                first["end_time"] = mid
                first["duration"] = mid - first["start_time"]
                # 先半分は Verse とする
                first["type"] = "verse"
                first["ascii_label"] = self.JP_ASCII_LABELS["verse"]
                # 後半は Pre-Chorus とする
                last["start_time"] = mid
                last["duration"] = sec["end_time"] - mid
                last["type"] = "pre_chorus"
                last["ascii_label"] = self.JP_ASCII_LABELS["pre_chorus"]
                final.extend([first, last])
            else:
                final.append(sec)
        enhanced = final

        # === ⑧ Solo → Instrumental consolidation =================
        merged2 = []
        for sec in enhanced:
            if sec["type"] == "solo" and sec["duration"] <= 6.0:
                # normalize
                sec["type"] = "instrumental"
                sec["ascii_label"] = self.JP_ASCII_LABELS["instrumental"]
            # merge consecutive instrumentals
            if (
                merged2
                and sec["type"] == "instrumental"
                and merged2[-1]["type"] == "instrumental"
            ):
                merged2[-1]["end_time"] = sec["end_time"]
                merged2[-1]["duration"] = (
                    merged2[-1]["end_time"] - merged2[-1]["start_time"]
                )
            else:
                merged2.append(sec)
        enhanced = merged2

        # === ⑨ too-short Chorus fix (<8bars) =====================
        min_chorus = (8 * 4 * 60) / bpm  # 8 bars
        patched = []
        for i, sec in enumerate(enhanced):
            if sec["type"] == "chorus" and sec["duration"] < min_chorus:
                # 優先: 直前が R なら吸収
                if patched and patched[-1]["type"] == "pre_chorus":
                    patched[-1]["end_time"] = sec["end_time"]
                    patched[-1]["duration"] = (
                        patched[-1]["end_time"] - patched[-1]["start_time"]
                    )
                    continue
                # さもなくば Instrumental 扱い
                sec["type"] = "instrumental"
                sec["ascii_label"] = self.JP_ASCII_LABELS["instrumental"]
            patched.append(sec)
        enhanced = patched

        # === ⑩ Advanced Verse/Pre-Chorus boundary correction =====
        corrected = []
        for i, sec in enumerate(enhanced):
            if sec["type"] == "verse" and i + 1 < len(enhanced):
                next_sec = enhanced[i + 1]
                # If verse is followed by chorus, consider converting to pre_chorus
                if next_sec["type"] == "chorus":
                    # Check energy level - if verse has higher energy than typical, convert to pre_chorus
                    if sec.get("energy_level", 0.0) > 0.55:
                        sec["type"] = "pre_chorus"
                        sec["ascii_label"] = self.JP_ASCII_LABELS["pre_chorus"]
            elif sec["type"] == "pre_chorus" and i > 0:
                prev_sec = enhanced[i - 1]
                # If pre_chorus follows another pre_chorus, convert first to verse
                if prev_sec["type"] == "pre_chorus":
                    prev_sec["type"] = "verse"
                    prev_sec["ascii_label"] = self.JP_ASCII_LABELS["verse"]
            corrected.append(sec)
        enhanced = corrected

        # Normalize instrumental aliases to standard "instrumental" type
        for sec in enhanced:
            if sec["type"] in self._INSTRUMENTAL_ALIASES:
                sec["type"] = "instrumental"
                sec["ascii_label"] = self.JP_ASCII_LABELS["instrumental"]

        # Re-run B-D-B merging after all processing
        enhanced = self._merge_chorus_instrumental_chorus(enhanced, bpm)

        # Final pass: Ensure all ASCII labels are consistent
        for section in enhanced:
            section['ascii_label'] = self.JP_ASCII_LABELS.get(
                section['type'], section['type']
            )

        return enhanced

    def _smart_merge_types(
        self, type1: str, type2: str, dur1: float, dur2: float
    ) -> str:
        """Smart merging of section types based on musical logic.

        Args:
            type1: First section type
            type2: Second section type
            dur1: Duration of first section
            dur2: Duration of second section

        Returns:
            Merged section type
        """
        # If one segment is much longer, prefer its type
        if dur1 > dur2 * 2:
            return type1
        elif dur2 > dur1 * 2:
            return type2

        # Special handling for short instrumental sections (likely fills/transitions)
        if type2 == 'instrumental' and dur2 < 8:  # Short instrumental
            # Merge short instrumental into previous section
            return type1
        elif type1 == 'instrumental' and dur1 < 8:  # Short instrumental
            # Merge short instrumental into following section
            return type2

        # Apply musical logic for merging
        merge_rules = {
            ('verse', 'pre_chorus'): 'verse',  # Pre-chorus often merges into verse
            ('pre_chorus', 'verse'): 'verse',
            ('pre_chorus', 'chorus'): 'chorus',  # Pre-chorus leads to chorus
            ('chorus', 'pre_chorus'): 'chorus',
            (
                'chorus',
                'instrumental',
            ): 'chorus',  # Short instrumental after chorus -> chorus
            (
                'instrumental',
                'chorus',
            ): 'chorus',  # Short instrumental before chorus -> chorus
            ('verse', 'bridge'): 'bridge',  # Bridge is more distinctive
            ('bridge', 'verse'): 'bridge',
            ('instrumental', 'break'): 'instrumental',
            ('break', 'instrumental'): 'instrumental',
            ('intro', 'verse'): 'verse',  # Intro usually leads to verse
            ('verse', 'outro'): 'outro',  # Outro is more distinctive
        }

        # Check both directions
        merged = merge_rules.get((type1, type2)) or merge_rules.get((type2, type1))
        if merged:
            return merged

        # Default: prefer the first type
        return type1

    def _merge_chorus_instrumental_chorus(
        self, sections: List[Dict[str, Any]], bpm: float
    ) -> List[Dict[str, Any]]:
        """
        Collapse patterns like B(8bars)-D(≤8bars)-B(8bars)
        into a single extended Chorus with maximum length limit.

        Args:
            sections: List of sections to process
            bpm: BPM for bar calculation

        Returns:
            Processed sections with B-D-B collapsed (max 16 bars per chorus)
        """
        if len(sections) < 3:
            return sections

        eight_bar = (8 * 4 * 60) / bpm  # 8 bars (=14.7s @130.5BPM)
        tol = 0.1 * eight_bar  # 10% tolerance (strict to prevent super-long chorus)
        MAX_BARS = 16  # Maximum 16 bars to prevent super-long sections

        changed = True
        while changed:  # Recursive/multi-pass for reliable merging
            changed = False
            out, i = [], 0
            while i < len(sections):
                if (
                    i + 2 < len(sections)
                    and sections[i]['type'] == 'chorus'
                    and sections[i + 1]['type'] == 'instrumental'
                    and sections[i + 2]['type'] == 'chorus'
                    and sections[i + 1]['duration'] <= eight_bar + tol
                ):

                    # Don't merge if instrumental was restored from consecutive chorus breaking
                    instrumental_section = sections[i + 1]
                    if instrumental_section.get('vocal_ratio') is not None:
                        # This instrumental was restored, don't merge it back
                        out.append(sections[i])
                        i += 1
                        continue

                    # Check merged length
                    merged_duration = (
                        sections[i + 2]['end_time'] - sections[i]['start_time']
                    )
                    max_allowed_duration = (
                        MAX_BARS * eight_bar / 8
                    )  # 16 bars equivalent time

                    if merged_duration <= max_allowed_duration:
                        # Execute merging within length limit
                        new_sec = sections[i].copy()
                        new_sec['end_time'] = sections[i + 2]['end_time']
                        new_sec['duration'] = merged_duration
                        # Always reset ASCII label after merge to prevent notation inconsistency
                        new_sec['ascii_label'] = self.JP_ASCII_LABELS.get(
                            new_sec['type'], new_sec['type']
                        )
                        out.append(new_sec)
                        i += 3
                        changed = True  # Loop again
                    else:
                        # Cancel merging if length limit exceeded, keep original 3 sections
                        out.append(sections[i])
                        i += 1
                else:
                    out.append(sections[i])
                    i += 1
            sections = out
        return sections

    def _downgrade_short_bridges(
        self, sections: List[Dict[str, Any]], bpm: float
    ) -> List[Dict[str, Any]]:
        """Downgrade short bridges (< 12 bars) to verse sections.

        Args:
            sections: List of sections to process
            bpm: BPM for bar calculation

        Returns:
            Processed sections with short bridges downgraded
        """
        if not sections:
            return sections

        # Calculate 12-bar duration threshold
        twelve_bar_duration = (
            12 * 4 * 60.0
        ) / bpm  # 12 bars * 4 beats/bar * 60s/min / bpm

        processed = sections.copy()

        for i, section in enumerate(processed):
            if section['type'] == 'bridge':
                duration = section.get(
                    'duration', section['end_time'] - section['start_time']
                )
                complexity = section.get('complexity', 0.0)
                energy_level = section.get('energy_level', 0.0)

                # Enhanced bridge filtering: duration ≥ 12 bars AND complexity > 0.7
                should_downgrade = duration < twelve_bar_duration or complexity <= 0.7

                if should_downgrade:
                    # Downgrade based on energy level
                    if energy_level > 0.55:
                        processed[i]['type'] = 'pre_chorus'
                        processed[i]['ascii_label'] = self.JP_ASCII_LABELS.get(
                            'pre_chorus', 'pre_chorus'
                        )
                    else:
                        processed[i]['type'] = 'verse'
                        processed[i]['ascii_label'] = self.JP_ASCII_LABELS.get(
                            'verse', 'verse'
                        )

        return processed

    def _cleanup_ending_instrumentals(
        self, sections: List[Dict[str, Any]], bpm: float = 130.0
    ) -> List[Dict[str, Any]]:
        """Clean up short instrumental sections at the end of the track.

        Args:
            sections: List of sections to process
            bpm: BPM for bar calculation

        Returns:
            Processed sections with ending instrumentals cleaned up
        """
        if not sections:
            return sections

        processed = sections.copy()

        # Convert ending instrumental or short verse to outro (≤8 bars)
        if processed:
            bar_sec = 4 * 60.0 / bpm  # Duration of 1 bar in seconds
            eight_bar_sec = 8 * bar_sec  # Duration of 8 bars
            last = processed[-1]

            last_dur = last.get('duration', last['end_time'] - last['start_time'])

            # Convert ending instrumental or short verse to outro (≤8 bars)
            if last['type'] in ['instrumental', 'verse'] and last_dur <= eight_bar_sec:
                last['type'] = 'outro'
                last['ascii_label'] = self.JP_ASCII_LABELS['outro']

        return processed

    # Delegate methods to analyzer
    def refine_section_labels_with_spectral_analysis(
        self, y: np.ndarray, sr: int, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine section labels using spectral flux analysis."""
        return self.analyzer.refine_section_labels_with_spectral_analysis(
            y, sr, sections
        )

    def analyze_form(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall musical form."""
        return self.analyzer.analyze_form(sections)

    def summarize_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Generate a summary text of sections for display."""
        return self.analyzer.summarize_sections(sections)

    def calculate_energy_scale(self, y: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive energy scale based on track characteristics."""
        return self.analyzer.calculate_energy_scale(y)

    # Backward compatibility methods (delegate to appropriate modules)
    def _section_to_letter(self, section_type: str) -> str:
        """Convert section type to letter for form notation."""
        return self.analyzer._section_to_letter(section_type)

    def _calculate_structural_complexity(self, sections: List[Dict[str, Any]]) -> float:
        """Calculate structural complexity score."""
        return self.analyzer._calculate_structural_complexity(sections)

    def _classify_instrumental_subtype(
        self, section: Dict[str, Any], spectral_features: Dict[str, Any] = None
    ) -> str:
        """Classify instrumental sections into more specific subtypes."""
        return self.analyzer.classify_instrumental_subtype(section, spectral_features)

    def _suppress_consecutive_pre_chorus(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suppress consecutive pre-chorus sections to prevent over-segmentation."""
        return self.jpop_optimizer.suppress_consecutive_pre_chorus(sections)

    def _enforce_pre_chorus_chorus_pairing(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enforce Pre-Chorus → Chorus pairing rules for J-Pop structure."""
        return self.jpop_optimizer.enforce_pre_chorus_chorus_pairing(sections)

    def _collapse_alternating_ar_patterns(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse A-R alternating patterns to A-R-B structure."""
        return self.jpop_optimizer.collapse_alternating_ar_patterns(sections)

    def _break_consecutive_chorus_chains(
        self, sections: List[Dict[str, Any]], y: np.ndarray, sr: int, bpm: float
    ) -> List[Dict[str, Any]]:
        """Break up consecutive chorus chains and restore instrumentals."""
        return self.jpop_optimizer.break_consecutive_chorus_chains(sections, y, sr, bpm)

    def _detect_vocal_presence(
        self, y: np.ndarray, sr: int, start_time: float, end_time: float
    ) -> float:
        """Detect vocal presence ratio in a given time segment."""
        return self.jpop_optimizer._detect_vocal_presence(y, sr, start_time, end_time)

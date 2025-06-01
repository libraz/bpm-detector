"""Music theory constants and utilities."""

# Note names for pitch analysis
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Progress callback step constants
PROGRESS_KEY_DETECTION = 20
PROGRESS_MELODY_EXTRACTION = 35
PROGRESS_MELODIC_ANALYSIS = 55
PROGRESS_HARMONIC_ANALYSIS = 75
PROGRESS_HARMONIC_RHYTHM = 90

# Interval names and their semitone distances
INTERVALS = {
    0: 'unison',
    1: 'minor_second',
    2: 'major_second',
    3: 'minor_third',
    4: 'major_third',
    5: 'perfect_fourth',
    6: 'tritone',
    7: 'perfect_fifth',
    8: 'minor_sixth',
    9: 'major_sixth',
    10: 'minor_seventh',
    11: 'major_seventh',
}

# Consonance ratings for intervals (0-1, higher = more consonant)
# Updated for modern pop music where 7ths and extended chords are common
CONSONANCE_RATINGS = {
    0: 1.0,  # unison
    1: 0.3,  # minor second (higher for modern music)
    2: 0.6,  # major second (higher for pop)
    3: 0.85,  # minor third (very common in pop)
    4: 0.95,  # major third (very consonant in pop)
    5: 0.9,  # perfect fourth (consonant)
    6: 0.6,  # tritone (much more accepted in modern music)
    7: 1.0,  # perfect fifth (most consonant)
    8: 0.8,  # minor sixth (common in pop)
    9: 0.85,  # major sixth (common in pop)
    10: 0.75,  # minor seventh (very common in pop/jazz)
    11: 0.7,  # major seventh (common in modern music)
}


def midi_to_note_name(midi_note: float) -> str:
    """Convert MIDI note number to note name with octave.

    Args:
        midi_note: MIDI note number

    Returns:
        Note name (e.g., 'C4', 'F#5')
    """
    import numpy as np

    if np.isnan(midi_note):
        return 'Unknown'

    note_number = int(round(midi_note))
    octave = (note_number // 12) - 1
    note_index = note_number % 12
    note_name = NOTE_NAMES[note_index]

    return f"{note_name}{octave}"


def classify_vocal_range(lowest: float, highest: float) -> str:
    """Classify vocal range based on lowest and highest notes.

    Args:
        lowest: Lowest MIDI note
        highest: Highest MIDI note

    Returns:
        Vocal range category
    """
    # Filter out extreme values that are likely instrumental
    if highest - lowest > 36:  # More than 3 octaves is likely instrumental
        return 'Instrumental (Wide Range)'

    if lowest < 36 or highest > 96:  # Below C2 or above C7 is likely instrumental
        return 'Instrumental (Extreme Range)'

    # Vocal range classifications (realistic MIDI note ranges)
    ranges = {
        'Bass': (41, 65),  # F2 to F4
        'Baritone': (45, 69),  # A2 to A4
        'Tenor': (48, 72),  # C3 to C5
        'Alto': (53, 77),  # F3 to F5
        'Mezzo-Soprano': (57, 81),  # A3 to A5
        'Soprano': (60, 84),  # C4 to C6
    }

    # Use the center of the range for classification, but also consider the lowest note
    center_note = (lowest + highest) / 2

    # If the lowest note is too low for female vocals, it's likely male or instrumental
    if lowest < 50:  # Below D3 is unlikely for female vocals
        # Focus on male vocal ranges
        male_ranges = ['Bass', 'Baritone', 'Tenor']
        best_match = 'Unknown'
        min_distance = float('inf')

        for range_name in male_ranges:
            range_low, range_high = ranges[range_name]
            range_center = (range_low + range_high) / 2
            distance = abs(center_note - range_center)

            overlap_low = max(lowest, range_low)
            overlap_high = min(highest, range_high)
            overlap_ratio = max(0, overlap_high - overlap_low) / (highest - lowest)

            if overlap_ratio > 0.3 and distance < min_distance:
                min_distance = distance
                best_match = range_name

        return best_match
    else:
        # Focus on female vocal ranges
        female_ranges = ['Alto', 'Mezzo-Soprano', 'Soprano']
        best_match = 'Unknown'
        min_distance = float('inf')

        for range_name in female_ranges:
            range_low, range_high = ranges[range_name]
            range_center = (range_low + range_high) / 2
            distance = abs(center_note - range_center)

            overlap_low = max(lowest, range_low)
            overlap_high = min(highest, range_high)
            overlap_ratio = max(0, overlap_high - overlap_low) / (highest - lowest)

            if overlap_ratio > 0.3 and distance < min_distance:
                min_distance = distance
                best_match = range_name

        return best_match if best_match != 'Unknown' else 'Alto'  # Default to Alto for female

"""Tests for CLI module."""

import io
import unittest
from unittest.mock import MagicMock, patch

from bpm_detector.cli import analyze_file_with_progress, main, print_results


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""

    def test_print_results_bpm_only(self):
        """Test printing results for BPM only analysis."""
        results = {
            "filename": "test.wav",
            "bpm": 120.5,
            "bpm_confidence": 85.3,
            "bpm_candidates": [(120.5, 45), (241.0, 23), (60.25, 18)],
        }

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            print_results(results, detect_key=False)

        output = captured_output.getvalue()

        # Check that essential information is present
        self.assertIn("test.wav", output)
        self.assertIn("120.50", output)  # Updated to match actual format
        self.assertIn("85.3%", output)
        self.assertIn("Estimated BPM", output)  # Updated to match actual output

    def test_print_results_with_key(self):
        """Test printing results with key detection."""
        results = {
            "filename": "test.wav",
            "bpm": 120.5,
            "bpm_confidence": 85.3,
            "bpm_candidates": [(120.5, 45), (241.0, 23)],
            "key": "C Major",
            "key_confidence": 78.9,
        }

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            print_results(results, detect_key=True)

        output = captured_output.getvalue()

        # Check that key information is present
        self.assertIn("C Major", output)
        self.assertIn("78.9%", output)
        self.assertIn("Estimated Key", output)

    @patch("bpm_detector.cli.AudioAnalyzer")
    @patch("soundfile.info")
    def test_analyze_file_success(self, mock_sf_info, mock_analyzer_class):
        """Test successful file analysis."""
        # Mock file info
        mock_info = MagicMock()
        mock_info.frames = 44100
        mock_sf_info.return_value = mock_info

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = {
            "filename": "test.wav",
            "bpm": 120.0,
            "bpm_confidence": 85.0,
            "bpm_candidates": [(120.0, 45)],
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Mock args
        args = MagicMock()
        args.sr = 22050
        args.detailed_progress = False
        args.detect_key = False
        args.min_bpm = 40.0
        args.max_bpm = 300.0
        args.start_bpm = 150.0

        # Capture stdout to avoid cluttering test output
        with patch("sys.stdout", io.StringIO()):
            analyze_file_with_progress("test.wav", mock_analyzer, args)

        # Verify analyzer was called correctly
        mock_analyzer.analyze_file.assert_called_once()
        call_args = mock_analyzer.analyze_file.call_args
        self.assertEqual(call_args[1]["path"], "test.wav")
        self.assertEqual(call_args[1]["detect_key"], False)

    @patch("bpm_detector.cli.AudioAnalyzer")
    @patch("soundfile.info")
    def test_analyze_file_with_error(self, mock_sf_info, mock_analyzer_class):
        """Test file analysis with error handling."""
        # Mock analyzer to raise an exception
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.side_effect = Exception("Test error")
        mock_analyzer_class.return_value = mock_analyzer

        # Mock file info
        mock_info = MagicMock()
        mock_info.frames = 44100
        mock_sf_info.return_value = mock_info

        # Mock args
        args = MagicMock()
        args.sr = 22050
        args.detailed_progress = False
        args.detect_key = False

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            analyze_file_with_progress("test.wav", mock_analyzer, args)

        output = captured_output.getvalue()

        # Should contain error message
        self.assertIn("Error processing", output)
        self.assertIn("test.wav", output)

    @patch("bpm_detector.cli.analyze_file_with_progress")
    @patch("os.path.exists")
    def test_main_single_file(self, mock_exists, mock_analyze):
        """Test main function with single file."""
        mock_exists.return_value = True

        test_args = ["bpm-detector", "test.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file_with_progress once
        self.assertEqual(mock_analyze.call_count, 1)

    @patch("bpm_detector.cli.SmartParallelAudioAnalyzer")
    @patch("os.path.exists")
    def test_main_multiple_files(self, mock_exists, mock_analyzer_class):
        """Test main function with multiple files."""
        mock_exists.return_value = True

        # Mock the analyzer instance
        mock_analyzer = MagicMock()
        mock_analyzer._parallel_config = MagicMock()
        mock_analyzer._parallel_config.enable_parallel = True

        # Mock both analyze_file and analyze_files methods
        mock_analyzer.analyze_file.return_value = {"basic_info": {"filename": "test.wav", "bpm": 120.0}}
        mock_analyzer.analyze_files.return_value = {
            "test1.wav": {"basic_info": {"filename": "test1.wav", "bpm": 120.0}},
            "test2.wav": {"basic_info": {"filename": "test2.wav", "bpm": 130.0}},
            "test3.wav": {"basic_info": {"filename": "test3.wav", "bpm": 140.0}},
        }
        mock_analyzer_class.return_value = mock_analyzer

        test_args = ["bpm-detector", "test1.wav", "test2.wav", "test3.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_files for multiple files or analyze_file for fallback
        total_calls = mock_analyzer.analyze_file.call_count + mock_analyzer.analyze_files.call_count
        self.assertGreater(total_calls, 0)
        # Verify that the analyzer was created
        mock_analyzer_class.assert_called_once()

    @patch("os.path.exists")
    def test_main_missing_file(self, mock_exists):
        """Test main function with missing file."""
        mock_exists.return_value = False

        test_args = ["bpm-detector", "missing.wav"]

        captured_output = io.StringIO()
        with patch("sys.argv", test_args):
            with patch("sys.stdout", captured_output):
                main()

        output = captured_output.getvalue()

        # Should show file not found message
        self.assertIn("File not found", output)
        self.assertIn("missing.wav", output)

    def test_main_with_options(self):
        """Test main function with various command line options."""
        test_args = [
            "bpm-detector",
            "--detect-key",
            "--detailed-progress",
            "--sr",
            "44100",
            "--min_bpm",
            "60",
            "--max_bpm",
            "200",
            "test.wav",
        ]

        with patch("sys.argv", test_args):
            with patch("bpm_detector.cli.analyze_file_with_progress") as mock_analyze:
                with patch("os.path.exists", return_value=True):
                    with patch("sys.stdout", io.StringIO()):
                        main()

        # Check that analyze_file_with_progress was called
        self.assertEqual(mock_analyze.call_count, 1)

        # Check the args passed to analyze_file_with_progress
        call_args = mock_analyze.call_args[0]
        args = call_args[2]  # Third argument is the args object

        self.assertTrue(args.detect_key)
        self.assertTrue(args.detailed_progress)
        self.assertEqual(args.sr, 44100)
        self.assertEqual(args.min_bpm, 60.0)
        self.assertEqual(args.max_bpm, 200.0)

    @patch("bpm_detector.cli.analyze_file_with_progress")
    @patch("os.path.exists")
    def test_main_quiet_option(self, mock_exists, mock_analyze):
        """Test that --quiet disables progress display."""
        mock_exists.return_value = True

        test_args = ["bpm-detector", "--quiet", "test.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file_with_progress once
        mock_analyze.assert_called_once()

        # Args should have progress disabled
        call_args = mock_analyze.call_args[0]
        args = call_args[2]
        self.assertFalse(args.progress)

    @patch("bpm_detector.cli.analyze_file_with_progress")
    @patch("os.path.exists")
    def test_main_selective_analysis_rhythm(self, mock_exists, mock_analyze):
        """Test selective analysis with --rhythm flag."""
        mock_exists.return_value = True

        test_args = ["bpm-detector", "--rhythm", "test.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file_with_progress once
        mock_analyze.assert_called_once()

        # Check that rhythm analysis is enabled
        call_args = mock_analyze.call_args[0]
        args = call_args[2]
        self.assertTrue(args.analyze_rhythm)
        self.assertFalse(args.analyze_chords)
        self.assertFalse(args.analyze_structure)
        self.assertFalse(args.analyze_timbre)
        self.assertFalse(args.analyze_melody)
        self.assertFalse(args.analyze_dynamics)

    @patch("bpm_detector.cli.analyze_file_with_progress")
    @patch("os.path.exists")
    def test_main_selective_analysis_multiple_flags(self, mock_exists, mock_analyze):
        """Test selective analysis with multiple flags."""
        mock_exists.return_value = True

        test_args = ["bpm-detector", "--rhythm", "--melody", "--timbre", "test.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file_with_progress once
        mock_analyze.assert_called_once()

        # Check that selected analyses are enabled
        call_args = mock_analyze.call_args[0]
        args = call_args[2]
        self.assertTrue(args.analyze_rhythm)
        self.assertTrue(args.analyze_melody)
        self.assertTrue(args.analyze_timbre)
        self.assertFalse(args.analyze_chords)
        self.assertFalse(args.analyze_structure)
        self.assertFalse(args.analyze_dynamics)

    @patch("bpm_detector.cli.analyze_file_with_progress")
    @patch("os.path.exists")
    def test_main_comprehensive_enables_all_analyses(self, mock_exists, mock_analyze):
        """Test that --comprehensive enables all analysis flags."""
        mock_exists.return_value = True

        test_args = ["bpm-detector", "--comprehensive", "test.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file_with_progress once
        mock_analyze.assert_called_once()

        # Check that all analyses are enabled
        call_args = mock_analyze.call_args[0]
        args = call_args[2]
        self.assertTrue(args.analyze_rhythm)
        self.assertTrue(args.analyze_chords)
        self.assertTrue(args.analyze_structure)
        self.assertTrue(args.analyze_timbre)
        self.assertTrue(args.analyze_melody)
        self.assertTrue(args.analyze_dynamics)

    def test_print_results_selective_rhythm(self):
        """Test printing results with selective rhythm analysis."""
        results = {
            "basic_info": {
                "filename": "test.wav",
                "bpm": 120.5,
                "bpm_confidence": 85.3,
                "bpm_candidates": [(120.5, 45), (241.0, 23)],
                "duration": 180.0,
                "key": None,
                "key_confidence": 0.0,
            },
            "rhythm": {"time_signature": "4/4", "groove_type": "straight"},
        }

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            print_results(results, detect_key=False, comprehensive=False)

        output = captured_output.getvalue()

        # Check that rhythm information is present
        self.assertIn("4/4", output)
        self.assertIn("straight", output)
        self.assertIn("Rhythm", output)

    def test_print_results_selective_multiple(self):
        """Test printing results with multiple selective analyses."""
        results = {
            "basic_info": {
                "filename": "test.wav",
                "bpm": 120.5,
                "bpm_confidence": 85.3,
                "bpm_candidates": [(120.5, 45)],
                "duration": 180.0,
                "key": "C Major",
                "key_confidence": 80.0,
            },
            "rhythm": {"time_signature": "4/4", "groove_type": "straight"},
            "timbre": {
                "dominant_instruments": [{"instrument": "piano", "confidence": 0.8}],
                "brightness": 0.7,
                "warmth": 0.6,
            },
        }

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            print_results(results, detect_key=True, comprehensive=False)

        output = captured_output.getvalue()

        # Check that selected analyses are present
        self.assertIn("Rhythm", output)
        self.assertIn("4/4", output)
        self.assertIn("Instruments", output)
        self.assertIn("piano", output)


if __name__ == "__main__":
    unittest.main()


def test_library_import_in_worker_preserves_sigint():
    import subprocess
    import sys

    script = '''
import signal
from concurrent.futures import ThreadPoolExecutor
handler = lambda signum, frame: None
signal.signal(signal.SIGINT, handler)
with ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(__import__, 'bpm_detector').result()
assert signal.getsignal(signal.SIGINT) is handler
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_sequential_cli_continues_after_corrupt_file(tmp_path, capsys):
    import numpy as np
    import soundfile as sf

    corrupt = tmp_path / 'corrupt.wav'
    corrupt.write_bytes(b'not audio')
    valid = tmp_path / 'valid.wav'
    sf.write(valid, np.zeros(22050, dtype=np.float32), 22050)
    with patch('sys.argv', ['bpm-detector', '--no-parallel', '--quiet', str(corrupt), str(valid)]):
        main()
    output = capsys.readouterr().out
    assert f'Error processing {corrupt}' in output
    assert 'valid.wav' in output
    assert 'Estimated BPM' in output


def test_cli_cleans_up_and_restores_signal_on_missing_file(tmp_path):
    import signal

    import bpm_detector.cli as cli

    before = signal.getsignal(signal.SIGINT)
    with patch('bpm_detector.cli.SmartParallelAudioAnalyzer') as analyzer_class:
        with patch('sys.argv', ['bpm-detector', str(tmp_path / 'missing.wav')]):
            main()
        analyzer_class.return_value.cleanup.assert_called_once()
    assert signal.getsignal(signal.SIGINT) is before
    assert cli.current_analyzer is None


def test_cli_modulation_requires_key():
    import pytest

    with patch('sys.argv', ['bpm-detector', '--detect-modulation', 'test.wav']):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_forwards_modulation_to_sequential_analyzer(tmp_path):
    path = tmp_path / 'file.wav'
    path.touch()
    with patch('bpm_detector.cli.AudioAnalyzer') as analyzer_class:
        with patch('bpm_detector.cli.print_results'):
            with patch(
                'sys.argv',
                ['bpm-detector', '--no-parallel', '--quiet', '--detect-key', '--detect-modulation', str(path)],
            ):
                main()
    assert analyzer_class.return_value.analyze_file.call_args.kwargs['detect_modulation'] is True


def test_cli_prints_modulation(capsys):
    results = {
        'basic_info': {'filename': 'x.wav', 'bpm': 120.0, 'bpm_confidence': 0.0},
        'modulation_analysis': {'modulations': [{'time': 12.0, 'from_key': 'C Major', 'to_key': 'D Major'}]},
    }
    print_results(results, detect_key=True)
    output = capsys.readouterr().out
    assert 'C Major' in output and 'D Major' in output
    assert '12.0' in output


def test_section_label_matching_type_is_not_repeated(capsys):
    sections = [
        {'type': 'intro', 'ascii_label': 'Intro', 'start_time': 0.0, 'duration': 4.0},
        {'type': 'verse', 'ascii_label': 'A-melo', 'start_time': 4.0, 'duration': 8.0},
    ]
    results = {
        'basic_info': {'filename': 'x.wav', 'bpm': 120.0, 'bpm_confidence': 0.0, 'duration': 12.0},
        'structure': {'form': 'IA', 'section_count': 2, 'sections': sections},
    }
    print_results(results)
    output = capsys.readouterr().out
    assert 'Intro(Intro)' not in output
    assert '1. Intro (00:00' in output
    assert 'Verse(A-melo)' in output


def test_cli_sigint_cleans_up_once_and_restores_handler():
    import signal

    import pytest

    import bpm_detector.cli as cli

    analyzer = MagicMock()
    previous = signal.getsignal(signal.SIGINT)

    def interrupt_cli():
        cli.current_analyzer = analyzer
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    with patch('bpm_detector.cli._main', interrupt_cli):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    analyzer.cleanup.assert_called_once()
    assert cli.current_analyzer is None
    assert signal.getsignal(signal.SIGINT) is previous

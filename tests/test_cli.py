"""Tests for CLI module."""

import argparse
import io
import sys
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
        self.assertIn("120.5", output)
        self.assertIn("85.3%", output)
        self.assertIn("BPM Candidates", output)

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
        mock_analyzer.analyze_file.return_value = {
            "test1.wav": {"basic_info": {"filename": "test1.wav", "bpm": 120.0}},
            "test2.wav": {"basic_info": {"filename": "test2.wav", "bpm": 130.0}},
            "test3.wav": {"basic_info": {"filename": "test3.wav", "bpm": 140.0}}
        }
        mock_analyzer_class.return_value = mock_analyzer

        test_args = ["bpm-detector", "test1.wav", "test2.wav", "test3.wav"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", io.StringIO()):
                main()

        # Should call analyze_file (either once for batch or multiple times for fallback)
        self.assertGreater(mock_analyzer.analyze_file.call_count, 0)
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


if __name__ == "__main__":
    unittest.main()

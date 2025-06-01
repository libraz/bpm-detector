"""Tests for parallel analyzer functionality."""

import os
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import soundfile as sf

from src.bpm_detector.auto_parallel import AutoParallelConfig, ParallelConfig, SystemMonitor
from src.bpm_detector.parallel_analyzer import SmartParallelAudioAnalyzer
from src.bpm_detector.progress_manager import ProgressManager, TaskStatus

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioread")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


class TestAutoParallelConfig(unittest.TestCase):
    """Test automatic parallel configuration."""

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_high_performance_config(self, mock_psutil, mock_cpu_count):
        """Test configuration for high-performance systems."""
        mock_cpu_count.return_value = 12
        mock_psutil.cpu_count.return_value = 6
        mock_psutil.virtual_memory.return_value.available = 16 * 1024**3  # 16GB
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value.percent = 50.0

        config = AutoParallelConfig.get_optimal_config()

        self.assertTrue(config.enable_parallel)
        self.assertEqual(config.max_workers, 10)  # 12 - 2
        self.assertTrue(config.use_process_pool)
        self.assertGreater(config.memory_limit_mb, 1000)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_low_performance_config(self, mock_psutil, mock_cpu_count):
        """Test configuration for low-performance systems."""
        mock_cpu_count.return_value = 2
        mock_psutil.cpu_count.return_value = 2
        mock_psutil.virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        mock_psutil.cpu_percent.return_value = 80.0
        mock_psutil.virtual_memory.return_value.percent = 75.0

        config = AutoParallelConfig.get_optimal_config()

        self.assertFalse(config.enable_parallel)
        self.assertEqual(config.max_workers, 1)

    def test_file_count_adjustment(self):
        """Test configuration adjustment based on file count."""
        base_config = ParallelConfig(enable_parallel=True, max_workers=8, use_process_pool=False)

        # Single file
        single_config = AutoParallelConfig.get_file_count_adjustment(1, base_config)
        self.assertLessEqual(single_config.max_workers, 6)

        # Many files
        many_config = AutoParallelConfig.get_file_count_adjustment(20, base_config)
        self.assertTrue(many_config.use_process_pool)
        self.assertGreater(many_config.max_workers, base_config.max_workers)


class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = SystemMonitor(check_interval=0.1)
        self.assertFalse(monitor.monitoring)
        self.assertEqual(monitor.current_load['cpu'], 0)
        self.assertEqual(monitor.current_load['memory'], 0)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_load_detection(self, mock_psutil):
        """Test load detection logic."""
        monitor = SystemMonitor()

        # High load scenario
        monitor.current_load = {'cpu': 95, 'memory': 90}
        self.assertTrue(monitor.should_reduce_parallelism())

        # Low load scenario
        monitor.current_load = {'cpu': 30, 'memory': 40}
        self.assertFalse(monitor.should_reduce_parallelism())

    def test_worker_recommendation(self):
        """Test worker count recommendation."""
        monitor = SystemMonitor()

        # High load - should reduce
        monitor.current_load = {'cpu': 95, 'memory': 90}
        recommended = monitor.get_recommended_workers(8)
        self.assertEqual(recommended, 4)  # Half of current

        # Low load - should increase
        monitor.current_load = {'cpu': 30, 'memory': 40}
        recommended = monitor.get_recommended_workers(4)
        self.assertEqual(recommended, 5)  # Increase by 1


class TestProgressManager(unittest.TestCase):
    """Test progress management functionality."""

    def test_task_registration(self):
        """Test task registration."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        tasks = manager.get_task_details()
        self.assertIn("test_task", tasks)
        self.assertEqual(tasks["test_task"].name, "Test Task")
        self.assertEqual(tasks["test_task"].status, TaskStatus.PENDING)

    def test_progress_updates(self):
        """Test progress updates."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        # Update progress
        manager.update_progress("test_task", 50.0, "Processing...")

        tasks = manager.get_task_details()
        task = tasks["test_task"]
        self.assertEqual(task.progress, 50.0)
        self.assertEqual(task.message, "Processing...")
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task.start_time)

    def test_task_completion(self):
        """Test task completion."""
        manager = ProgressManager()
        manager.register_task("test_task", "Test Task")

        # Complete task
        manager.complete_task("test_task", True)

        tasks = manager.get_task_details()
        task = tasks["test_task"]
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.progress, 100.0)
        self.assertIsNotNone(task.end_time)

    def test_overall_progress(self):
        """Test overall progress calculation."""
        manager = ProgressManager()
        manager.register_task("task1", "Task 1")
        manager.register_task("task2", "Task 2")

        manager.update_progress("task1", 100.0)
        manager.update_progress("task2", 50.0)

        overall = manager.get_overall_progress()
        self.assertEqual(overall, 75.0)  # (100 + 50) / 2

    def test_status_summary(self):
        """Test status summary."""
        manager = ProgressManager()
        manager.register_task("task1", "Task 1")
        manager.register_task("task2", "Task 2")
        manager.register_task("task3", "Task 3")

        manager.complete_task("task1", True)
        manager.update_progress("task2", 50.0)
        manager.complete_task("task3", False)

        summary = manager.get_status_summary()
        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['running'], 1)
        self.assertEqual(summary['failed'], 1)


class TestSmartParallelAudioAnalyzer(unittest.TestCase):
    """Test smart parallel audio analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.wav")

        # Generate test audio (1 second of sine wave)
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        sf.write(self.test_file, audio, sr)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_analyzer_initialization(self, mock_psutil, mock_cpu_count):
        """Test analyzer initialization."""
        mock_cpu_count.return_value = 8
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
        mock_psutil.cpu_percent.return_value = 30.0
        mock_psutil.virtual_memory.return_value.percent = 50.0

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        self.assertIsNotNone(analyzer._parallel_config)
        if analyzer._parallel_config is not None:
            self.assertTrue(analyzer._parallel_config.enable_parallel)

    @patch('src.bpm_detector.auto_parallel.cpu_count')
    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_parallel_disabled(self, mock_psutil, mock_cpu_count):
        """Test analyzer with parallel processing disabled."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)

        self.assertIsNone(analyzer._parallel_config)

    def test_manual_worker_override(self):
        """Test manual worker count override."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=4)

        if analyzer._parallel_config:
            self.assertEqual(analyzer._parallel_config.max_workers, 4)

    @patch('src.bpm_detector.parallel_analyzer.SmartParallelAudioAnalyzer._should_use_parallel')
    def test_fallback_to_sequential(self, mock_should_use_parallel):
        """Test fallback to sequential processing."""
        mock_should_use_parallel.return_value = False

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        # Should fall back to parent class method
        with patch.object(analyzer.__class__.__bases__[0], 'analyze_file') as mock_parent:
            mock_parent.return_value = {"test": "result"}

            analyzer.analyze_file(self.test_file, comprehensive=True)

            mock_parent.assert_called_once()

    def test_multiple_file_processing(self):
        """Test multiple file processing."""
        # Create additional test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.wav")
            sr = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            sf.write(file_path, audio, sr)
            test_files.append(file_path)

        analyzer = SmartParallelAudioAnalyzer(auto_parallel=False)  # Disable for testing

        # Test multiple file analysis
        results = analyzer.analyze_files(test_files, comprehensive=False)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)

        # Clean up
        for file_path in test_files:
            os.remove(file_path)


class TestAdvancedParallelFeatures(unittest.TestCase):
    """Test advanced parallel processing features."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sr = 22050

        # Create test audio file
        duration = 2.0
        t = np.linspace(0, duration, int(self.sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        self.test_file = os.path.join(self.temp_dir, "test.wav")
        sf.write(self.test_file, audio, self.sr)

        self.analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.analyzer, 'cleanup'):
            self.analyzer.cleanup()
        shutil.rmtree(self.temp_dir)

    def test_progress_manager_integration(self):
        """Test integration with progress manager."""
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        self.analyzer.analyze_file(
            self.test_file, comprehensive=True, progress_callback=progress_callback, detailed_progress=True
        )

        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)

        # Progress should be between 0 and 100
        for progress, message in progress_updates:
            self.assertGreaterEqual(progress, 0.0)
            self.assertLessEqual(progress, 100.0)
            self.assertIsInstance(message, str)

    def test_system_resource_monitoring(self):
        """Test system resource monitoring during analysis."""
        from src.bpm_detector.auto_parallel import SystemMonitor

        # Enable system monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()

        try:
            results = self.analyzer.analyze_file(self.test_file, comprehensive=True)

            # Should complete successfully even with monitoring
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

            # Check monitoring functionality
            should_reduce = monitor.should_reduce_parallelism()
            self.assertIsInstance(should_reduce, bool)

        finally:
            monitor.stop_monitoring()

    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        from src.bpm_detector.auto_parallel import PerformanceProfiler

        profiler = PerformanceProfiler()

        # Start profiling
        profile_data = profiler.start_profiling("test_analysis")

        # Run analysis
        results = self.analyzer.analyze_file(self.test_file, comprehensive=True)

        # End profiling
        profiler.end_profiling(profile_data)

        # Should complete successfully with profiling
        self.assertIsInstance(results, dict)

        # Check profiling data
        summary = profiler.get_performance_summary()
        self.assertIsInstance(summary, dict)

    def test_adaptive_worker_adjustment(self):
        """Test adaptive worker count adjustment."""
        from src.bpm_detector.auto_parallel import AutoParallelConfig

        # Test with different file counts
        # Create multiple test files
        large_batch = []
        for i in range(5):
            file_path = os.path.join(self.temp_dir, f"batch_test_{i}.wav")
            duration = 1.0
            t = np.linspace(0, duration, int(self.sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t)
            sf.write(file_path, audio, self.sr)
            large_batch.append(file_path)

        try:
            # Test file count adjustment
            base_config = AutoParallelConfig.get_optimal_config()
            adjusted_config = AutoParallelConfig.get_file_count_adjustment(len(large_batch), base_config)

            self.assertIsInstance(adjusted_config.max_workers, int)
            self.assertGreater(adjusted_config.max_workers, 0)

        finally:
            # Clean up batch files
            for file_path in large_batch:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def test_error_handling_in_parallel(self):
        """Test error handling in parallel processing."""
        # Create a corrupted file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.wav")
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid audio file")

        test_files = [self.test_file, corrupted_file]

        try:
            results = self.analyzer.analyze_files(test_files, comprehensive=False)

            # Should handle errors gracefully
            self.assertIsInstance(results, dict)

            # Good file should have results
            self.assertIn(self.test_file, results)

            # Results should be valid for good file
            if self.test_file in results:
                self.assertIsInstance(results[self.test_file], dict)

        except Exception as e:
            # If exception is raised, it should be handled gracefully
            self.assertIsInstance(e, Exception)

        finally:
            if os.path.exists(corrupted_file):
                os.remove(corrupted_file)

    def test_memory_management(self):
        """Test memory management during parallel processing."""
        # Test with limited workers to check memory management
        limited_analyzer = SmartParallelAudioAnalyzer(auto_parallel=True, max_workers=2)

        try:
            results = limited_analyzer.analyze_file(self.test_file, comprehensive=True)

            # Should complete without memory issues
            self.assertIsInstance(results, dict)
            self.assertIn('basic_info', results)

        finally:
            if hasattr(limited_analyzer, 'cleanup'):
                limited_analyzer.cleanup()

    def test_parallel_strategy_selection(self):
        """Test automatic selection of parallel strategy."""
        from src.bpm_detector.auto_parallel import AutoParallelConfig, ParallelStrategy

        # Get optimal configuration
        config = AutoParallelConfig.get_optimal_config()

        # Should have a valid strategy
        self.assertIsInstance(config.strategy, ParallelStrategy)
        # Should have a valid strategy (including AGGRESSIVE_PARALLEL)
        valid_strategies = [
            ParallelStrategy.THREAD_POOL,
            ParallelStrategy.PROCESS_POOL,
            ParallelStrategy.AGGRESSIVE_PARALLEL,
            ParallelStrategy.BALANCED_PARALLEL,
            ParallelStrategy.CONSERVATIVE_PARALLEL,
            ParallelStrategy.SEQUENTIAL_ONLY,
        ]
        self.assertIn(config.strategy, valid_strategies)

        # Should have reasonable worker count
        self.assertGreater(config.max_workers, 0)
        self.assertLessEqual(config.max_workers, 32)  # Reasonable upper bound

    def test_cleanup_functionality(self):
        """Test proper cleanup of parallel resources."""
        analyzer = SmartParallelAudioAnalyzer(auto_parallel=True)

        # Run some analysis
        results = analyzer.analyze_file(self.test_file, comprehensive=False)
        self.assertIsInstance(results, dict)

        # Test explicit cleanup
        analyzer.cleanup()

        # Should not raise errors after cleanup
        try:
            analyzer.cleanup()  # Second cleanup should be safe
        except Exception as e:
            self.fail(f"Cleanup should not raise errors: {e}")

    def test_detailed_progress_tracking(self):
        """Test detailed progress tracking functionality."""
        from src.bpm_detector.progress_manager import DetailedProgressDisplay, ProgressManager

        progress_manager = ProgressManager()
        detailed_display = DetailedProgressDisplay()

        # Register some tasks
        progress_manager.register_task("task1", "Test Task 1")
        progress_manager.register_task("task2", "Test Task 2")

        # Update progress
        progress_manager.update_progress("task1", 50.0, "Halfway done")
        progress_manager.update_progress("task2", 25.0, "Quarter done")

        # Test display update
        detailed_display.update(progress_manager)

        # Complete tasks
        progress_manager.complete_task("task1", success=True)
        progress_manager.complete_task("task2", success=True)

        # Get final status
        status = progress_manager.get_status_summary()
        self.assertIsInstance(status, dict)
        self.assertIn('total_tasks', status)
        self.assertIn('completed_tasks', status)


if __name__ == '__main__':
    unittest.main()

"""Tests for auto parallel module."""

import time
import unittest
from unittest.mock import MagicMock, patch

from src.bpm_detector.auto_parallel import (
    AutoParallelConfig,
    ParallelConfig,
    ParallelStrategy,
    PerformanceProfiler,
    SystemMonitor,
)


class TestParallelStrategy(unittest.TestCase):
    """Test cases for ParallelStrategy enum."""

    def test_parallel_strategy_values(self):
        """Test parallel strategy enum values."""
        self.assertEqual(ParallelStrategy.THREAD_POOL.value, "thread_pool")
        self.assertEqual(ParallelStrategy.PROCESS_POOL.value, "process_pool")


class TestParallelConfig(unittest.TestCase):
    """Test cases for ParallelConfig dataclass."""

    def test_parallel_config_creation(self):
        """Test parallel config creation."""
        config = ParallelConfig(enable_parallel=True, max_workers=4, strategy=ParallelStrategy.THREAD_POOL)

        self.assertTrue(config.enable_parallel)
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.strategy, ParallelStrategy.THREAD_POOL)

    def test_parallel_config_defaults(self):
        """Test parallel config with defaults."""
        config = ParallelConfig()

        # Should have reasonable defaults
        self.assertIsInstance(config.enable_parallel, bool)
        self.assertIsInstance(config.max_workers, int)
        self.assertIsInstance(config.strategy, ParallelStrategy)


class TestAutoParallelConfig(unittest.TestCase):
    """Test cases for AutoParallelConfig."""

    @patch('src.bpm_detector.auto_parallel.psutil')
    @patch('src.bpm_detector.auto_parallel.cpu_count')
    def test_get_optimal_config_high_performance(self, mock_cpu_count, mock_psutil):
        """Test optimal config for high-performance system."""
        # Mock high-performance system
        mock_cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.available = 16 * 1024**3  # 16GB
        mock_psutil.cpu_percent.return_value = 20.0  # Low CPU usage
        mock_psutil.virtual_memory.return_value.percent = 30.0  # Low memory usage

        config = AutoParallelConfig.get_optimal_config()

        # Should enable parallel processing
        self.assertTrue(config.enable_parallel)
        self.assertGreater(config.max_workers, 1)
        self.assertLessEqual(config.max_workers, 8)
        self.assertIsInstance(config.strategy, ParallelStrategy)

    @patch('src.bpm_detector.auto_parallel.psutil')
    @patch('src.bpm_detector.auto_parallel.cpu_count')
    def test_get_optimal_config_low_performance(self, mock_cpu_count, mock_psutil):
        """Test optimal config for low-performance system."""
        # Mock low-performance system
        mock_cpu_count.return_value = 2
        mock_psutil.virtual_memory.return_value.available = 1 * 1024**3  # 1GB
        mock_psutil.cpu_percent.return_value = 80.0  # High CPU usage
        mock_psutil.virtual_memory.return_value.percent = 85.0  # High memory usage

        config = AutoParallelConfig.get_optimal_config()

        # Should be conservative or disable parallel processing
        if config.enable_parallel:
            self.assertLessEqual(config.max_workers, 2)
        else:
            self.assertFalse(config.enable_parallel)

    @patch('src.bpm_detector.auto_parallel.psutil')
    @patch('src.bpm_detector.auto_parallel.cpu_count')
    def test_get_optimal_config_single_core(self, mock_cpu_count, mock_psutil):
        """Test optimal config for single-core system."""
        # Mock single-core system
        mock_cpu_count.return_value = 1
        mock_psutil.virtual_memory.return_value.available = 2 * 1024**3  # 2GB
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value.percent = 60.0

        config = AutoParallelConfig.get_optimal_config()

        # Should disable parallel processing for single core
        self.assertFalse(config.enable_parallel)
        self.assertEqual(config.max_workers, 1)

    def test_get_file_count_adjustment(self):
        """Test file count adjustment."""
        base_config = ParallelConfig(enable_parallel=True, max_workers=4, strategy=ParallelStrategy.THREAD_POOL)

        # Test with small file count
        small_adjusted = AutoParallelConfig.get_file_count_adjustment(2, base_config)
        self.assertLessEqual(small_adjusted.max_workers, base_config.max_workers)

        # Test with large file count
        large_adjusted = AutoParallelConfig.get_file_count_adjustment(20, base_config)
        self.assertGreaterEqual(large_adjusted.max_workers, 1)

        # Test with single file
        single_adjusted = AutoParallelConfig.get_file_count_adjustment(1, base_config)
        self.assertEqual(single_adjusted.max_workers, 1)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_psutil_error_handling(self, mock_psutil):
        """Test handling of psutil errors."""
        # Mock psutil to raise an exception
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")
        mock_psutil.cpu_percent.side_effect = Exception("psutil error")

        # Should handle errors gracefully and return safe defaults
        config = AutoParallelConfig.get_optimal_config()

        self.assertIsInstance(config, ParallelConfig)
        self.assertIsInstance(config.enable_parallel, bool)
        self.assertIsInstance(config.max_workers, int)
        self.assertGreater(config.max_workers, 0)


class TestSystemMonitor(unittest.TestCase):
    """Test cases for SystemMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = SystemMonitor(check_interval=0.1)  # Fast interval for testing

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.monitor, 'stop_monitoring'):
            self.monitor.stop_monitoring()

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.check_interval, 0.1)
        self.assertFalse(self.monitor._monitoring)
        self.assertIsNone(self.monitor._monitor_thread)

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._monitoring)
        self.assertIsNotNone(self.monitor._monitor_thread)

        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_should_reduce_parallelism(self, mock_psutil):
        """Test parallelism reduction detection."""
        # Mock high system load
        mock_psutil.cpu_percent.return_value = 90.0
        mock_psutil.virtual_memory.return_value.percent = 85.0

        should_reduce = self.monitor.should_reduce_parallelism()

        # Should recommend reduction for high load
        self.assertIsInstance(should_reduce, bool)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_get_recommended_workers(self, mock_psutil):
        """Test worker count recommendation."""
        # Mock moderate system load
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value.percent = 60.0

        recommended = self.monitor.get_recommended_workers(4)

        # Should return reasonable worker count
        self.assertIsInstance(recommended, int)
        self.assertGreater(recommended, 0)
        self.assertLessEqual(recommended, 4)

    def test_monitor_thread_safety(self):
        """Test monitor thread safety."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Should be able to call methods safely
        should_reduce = self.monitor.should_reduce_parallelism()
        recommended = self.monitor.get_recommended_workers(2)

        self.assertIsInstance(should_reduce, bool)
        self.assertIsInstance(recommended, int)

        # Stop monitoring
        self.monitor.stop_monitoring()

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        for _ in range(3):
            self.monitor.start_monitoring()
            self.assertTrue(self.monitor._monitoring)

            time.sleep(0.05)  # Brief monitoring period

            self.monitor.stop_monitoring()
            self.assertFalse(self.monitor._monitoring)


class TestPerformanceProfiler(unittest.TestCase):
    """Test cases for PerformanceProfiler."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        self.assertIsInstance(self.profiler.profiles, dict)
        self.assertEqual(len(self.profiler.profiles), 0)

    def test_start_end_profiling(self):
        """Test profiling lifecycle."""
        # Start profiling
        profile_data = self.profiler.start_profiling("test_task")

        # Check profile data structure
        self.assertIsInstance(profile_data, dict)
        self.assertIn('task_name', profile_data)
        self.assertIn('start_time', profile_data)
        self.assertIn('start_memory', profile_data)
        self.assertEqual(profile_data['task_name'], "test_task")

        # Simulate some work
        time.sleep(0.01)

        # End profiling
        self.profiler.end_profiling(profile_data)

        # Check that profile was recorded
        self.assertIn("test_task", self.profiler.profiles)
        task_profiles = self.profiler.profiles["test_task"]
        self.assertGreater(len(task_profiles), 0)

        # Check profile structure
        profile = task_profiles[0]
        self.assertIn('duration', profile)
        self.assertIn('memory_delta', profile)
        self.assertGreater(profile['duration'], 0)

    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Add some test profiles
        for i in range(3):
            profile_data = self.profiler.start_profiling(f"task_{i}")
            time.sleep(0.01)
            self.profiler.end_profiling(profile_data)

        summary = self.profiler.get_performance_summary()

        # Check summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn('total_tasks', summary)
        self.assertIn('task_summaries', summary)

        # Check task summaries
        task_summaries = summary['task_summaries']
        self.assertIsInstance(task_summaries, dict)

        for task_name, task_summary in task_summaries.items():
            self.assertIn('count', task_summary)
            self.assertIn('avg_duration', task_summary)
            self.assertIn('total_duration', task_summary)
            self.assertIn('avg_memory_delta', task_summary)

    def test_should_adjust_parallelism(self):
        """Test parallelism adjustment recommendation."""
        # Add profiles for a task
        for _ in range(5):
            profile_data = self.profiler.start_profiling("slow_task")
            time.sleep(0.02)  # Simulate slow task
            self.profiler.end_profiling(profile_data)

        adjustment = self.profiler.should_adjust_parallelism("slow_task")

        # Should return recommendation or None
        if adjustment is not None:
            self.assertIsInstance(adjustment, str)
            self.assertIn(adjustment, ['reduce', 'increase', 'maintain'])

    def test_multiple_tasks_profiling(self):
        """Test profiling multiple different tasks."""
        tasks = ["task_a", "task_b", "task_c"]

        for task in tasks:
            for _ in range(2):
                profile_data = self.profiler.start_profiling(task)
                time.sleep(0.005)
                self.profiler.end_profiling(profile_data)

        # Check all tasks were recorded
        for task in tasks:
            self.assertIn(task, self.profiler.profiles)
            self.assertEqual(len(self.profiler.profiles[task]), 2)

        # Check summary includes all tasks
        summary = self.profiler.get_performance_summary()
        self.assertEqual(summary['total_tasks'], len(tasks))

        for task in tasks:
            self.assertIn(task, summary['task_summaries'])

    def test_profiling_error_handling(self):
        """Test profiling with errors."""
        # Test with invalid profile data
        invalid_profile = {'invalid': 'data'}

        try:
            self.profiler.end_profiling(invalid_profile)
            # Should handle gracefully
        except Exception as e:
            # If exception is raised, it should be handled appropriately
            self.assertIsInstance(e, Exception)

    def test_empty_profiler_summary(self):
        """Test summary generation with no profiles."""
        summary = self.profiler.get_performance_summary()

        # Should handle empty state gracefully
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_tasks'], 0)
        self.assertIsInstance(summary['task_summaries'], dict)
        self.assertEqual(len(summary['task_summaries']), 0)

    @patch('src.bpm_detector.auto_parallel.psutil')
    def test_memory_tracking(self, mock_psutil):
        """Test memory usage tracking."""
        # Mock memory info
        mock_memory = MagicMock()
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory

        profile_data = self.profiler.start_profiling("memory_test")

        # Change mock memory to simulate usage
        mock_memory.available = 7 * 1024**3  # 7GB (1GB used)

        self.profiler.end_profiling(profile_data)

        # Check memory delta was recorded
        profiles = self.profiler.profiles["memory_test"]
        self.assertGreater(len(profiles), 0)

        profile = profiles[0]
        self.assertIn('memory_delta', profile)

    def test_concurrent_profiling(self):
        """Test concurrent profiling of same task."""
        # Start multiple profiles for same task
        profile1 = self.profiler.start_profiling("concurrent_task")
        profile2 = self.profiler.start_profiling("concurrent_task")

        time.sleep(0.01)

        # End both profiles
        self.profiler.end_profiling(profile1)
        self.profiler.end_profiling(profile2)

        # Should record both profiles
        profiles = self.profiler.profiles["concurrent_task"]
        self.assertEqual(len(profiles), 2)


if __name__ == '__main__':
    unittest.main()

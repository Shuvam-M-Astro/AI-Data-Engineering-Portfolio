#!/usr/bin/env python3
"""
Debug Tools for IMDB RPA Sentiment Analysis Tool

This module provides various debugging and monitoring utilities to help troubleshoot
and optimize the IMDB sentiment analysis tool.
"""

import time
import psutil
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    processing_time: float
    movies_processed: int
    reviews_processed: int
    errors_count: int

class PerformanceMonitor:
    """Monitors system performance during execution."""
    
    def __init__(self, output_dir: str = "debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def record_metrics(self, movies_processed: int = 0, reviews_processed: int = 0, errors_count: int = 0):
        """Record current performance metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network I/O
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Calculate processing time
            processing_time = time.time() - self.start_time if self.start_time else 0
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_io=network_data,
                processing_time=processing_time,
                movies_processed=movies_processed,
                reviews_processed=reviews_processed,
                errors_count=errors_count
            )
            
            self.metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
    
    def save_metrics(self, filename: str = None):
        """Save performance metrics to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump([asdict(metric) for metric in self.metrics], f, indent=2)
            logger.info(f"Performance metrics saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def generate_performance_report(self) -> str:
        """Generate a performance report."""
        if not self.metrics:
            return "No performance metrics available."
        
        latest = self.metrics[-1]
        
        report = f"""
Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

System Performance:
- CPU Usage: {latest.cpu_percent:.1f}%
- Memory Usage: {latest.memory_percent:.1f}% ({latest.memory_used_mb:.1f} MB)
- Disk Usage: {latest.disk_usage_percent:.1f}%
- Total Processing Time: {latest.processing_time:.2f} seconds

Processing Statistics:
- Movies Processed: {latest.movies_processed}
- Reviews Processed: {latest.reviews_processed}
- Errors Encountered: {latest.errors_count}

Network Activity:
- Bytes Sent: {latest.network_io['bytes_sent']:,}
- Bytes Received: {latest.network_io['bytes_recv']:,}
- Packets Sent: {latest.network_io['packets_sent']:,}
- Packets Received: {latest.network_io['packets_recv']:,}
"""
        return report

class DebugLogger:
    """Enhanced logging for debugging purposes."""
    
    def __init__(self, log_file: str = "debug.log", level: str = "DEBUG"):
        self.log_file = log_file
        self.level = getattr(logging, level.upper())
        
        # Configure debug logger
        self.logger = logging.getLogger("debug")
        self.logger.setLevel(self.level)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_function_call(self, func_name: str, args: tuple, kwargs: dict):
        """Log function calls with arguments."""
        self.logger.debug(f"Function call: {func_name}(args={args}, kwargs={kwargs})")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exceptions with full traceback."""
        self.logger.error(f"Exception in {context}: {exception}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

@contextmanager
def debug_context(context_name: str, debug_logger: DebugLogger):
    """Context manager for debugging function execution."""
    start_time = time.time()
    debug_logger.logger.debug(f"Entering context: {context_name}")
    
    try:
        yield
    except Exception as e:
        debug_logger.log_exception(e, context_name)
        raise
    finally:
        execution_time = time.time() - start_time
        debug_logger.logger.debug(f"Exiting context: {context_name} (took {execution_time:.2f}s)")

class NetworkMonitor:
    """Monitors network activity and connectivity."""
    
    def __init__(self):
        self.initial_network_stats = None
        self.final_network_stats = None
    
    def start_monitoring(self):
        """Start network monitoring."""
        self.initial_network_stats = psutil.net_io_counters()
        logger.info("Network monitoring started")
    
    def stop_monitoring(self):
        """Stop network monitoring and calculate statistics."""
        self.final_network_stats = psutil.net_io_counters()
        
        if self.initial_network_stats:
            bytes_sent = self.final_network_stats.bytes_sent - self.initial_network_stats.bytes_sent
            bytes_recv = self.final_network_stats.bytes_recv - self.initial_network_stats.bytes_recv
            
            logger.info(f"Network activity: {bytes_sent:,} bytes sent, {bytes_recv:,} bytes received")
    
    def check_connectivity(self, url: str = "https://www.imdb.com") -> bool:
        """Check connectivity to a specific URL."""
        try:
            import requests
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connectivity check failed for {url}: {e}")
            return False

class MemoryProfiler:
    """Profiles memory usage during execution."""
    
    def __init__(self):
        self.memory_snapshots = []
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
        
        self.memory_snapshots.append(snapshot)
        logger.debug(f"Memory snapshot '{label}': {snapshot['rss_mb']:.2f} MB")
    
    def generate_memory_report(self) -> str:
        """Generate a memory usage report."""
        if not self.memory_snapshots:
            return "No memory snapshots available."
        
        report = "Memory Usage Report\n"
        report += "=" * 50 + "\n"
        
        for snapshot in self.memory_snapshots:
            report += f"{snapshot['timestamp']} - {snapshot['label']}: "
            report += f"{snapshot['rss_mb']:.2f} MB ({snapshot['percent']:.1f}%)\n"
        
        return report

class ErrorAnalyzer:
    """Analyzes and categorizes errors."""
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, error: Exception, context: str, severity: str = "ERROR"):
        """Add an error to the analysis."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'severity': severity,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        logger.error(f"Error in {context}: {error}")
    
    def categorize_errors(self) -> Dict[str, List[Dict]]:
        """Categorize errors by type."""
        categories = {}
        
        for error in self.errors:
            error_type = error['error_type']
            if error_type not in categories:
                categories[error_type] = []
            categories[error_type].append(error)
        
        return categories
    
    def generate_error_report(self) -> str:
        """Generate an error analysis report."""
        if not self.errors:
            return "No errors recorded."
        
        categories = self.categorize_errors()
        
        report = "Error Analysis Report\n"
        report += "=" * 50 + "\n"
        report += f"Total Errors: {len(self.errors)}\n\n"
        
        for error_type, error_list in categories.items():
            report += f"{error_type} ({len(error_list)} occurrences):\n"
            for error in error_list:
                report += f"  - {error['context']}: {error['error_message']}\n"
            report += "\n"
        
        return report

class DebugTools:
    """Main debug tools class that combines all debugging utilities."""
    
    def __init__(self, output_dir: str = "debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.performance_monitor = PerformanceMonitor(str(self.output_dir))
        self.debug_logger = DebugLogger(str(self.output_dir / "debug.log"))
        self.network_monitor = NetworkMonitor()
        self.memory_profiler = MemoryProfiler()
        self.error_analyzer = ErrorAnalyzer()
    
    def start_debugging(self):
        """Start all debugging tools."""
        self.performance_monitor.start_monitoring()
        self.network_monitor.start_monitoring()
        self.memory_profiler.take_snapshot("start")
        self.debug_logger.logger.info("Debug tools started")
    
    def stop_debugging(self):
        """Stop all debugging tools and generate reports."""
        self.network_monitor.stop_monitoring()
        self.memory_profiler.take_snapshot("end")
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance report
        perf_report = self.performance_monitor.generate_performance_report()
        with open(self.output_dir / f"performance_report_{timestamp}.txt", 'w') as f:
            f.write(perf_report)
        
        # Memory report
        memory_report = self.memory_profiler.generate_memory_report()
        with open(self.output_dir / f"memory_report_{timestamp}.txt", 'w') as f:
            f.write(memory_report)
        
        # Error report
        error_report = self.error_analyzer.generate_error_report()
        with open(self.output_dir / f"error_report_{timestamp}.txt", 'w') as f:
            f.write(error_report)
        
        # Save performance metrics
        self.performance_monitor.save_metrics(f"performance_metrics_{timestamp}.json")
        
        self.debug_logger.logger.info("Debug tools stopped and reports generated")

def main():
    """Main function for standalone debug tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug tools for IMDB sentiment analysis")
    parser.add_argument("--check-connectivity", action="store_true", help="Check network connectivity")
    parser.add_argument("--system-info", action="store_true", help="Display system information")
    parser.add_argument("--memory-profile", action="store_true", help="Run memory profiling")
    
    args = parser.parse_args()
    
    if args.check_connectivity:
        monitor = NetworkMonitor()
        if monitor.check_connectivity():
            print("✓ Connectivity to IMDB is working")
        else:
            print("✗ Connectivity to IMDB failed")
    
    if args.system_info:
        print("System Information:")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        print(f"Disk: {psutil.disk_usage('/').total / 1024 / 1024 / 1024:.1f} GB")
    
    if args.memory_profile:
        profiler = MemoryProfiler()
        profiler.take_snapshot("initial")
        
        # Simulate some work
        import time
        time.sleep(2)
        
        profiler.take_snapshot("after_work")
        print(profiler.generate_memory_report())

if __name__ == "__main__":
    main() 
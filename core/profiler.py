import time
import torch
from collections import defaultdict, deque
import numpy as np


class PerformanceProfiler:

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = defaultdict(list)  # section_name -> list of times
        self.call_counts = defaultdict(int)
        self.active_sections = {}  # section_name -> start_time
        self.step_count = 0

        # Rolling window for recent performance
        self.recent_window = 1000
        self.recent_timings = defaultdict(lambda: deque(maxlen=self.recent_window))

    def start(self, section_name):
        if not self.enabled:
            return

        # Handle CUDA synchronization for accurate GPU timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.active_sections[section_name] = time.perf_counter()

    def end(self, section_name):
        if not self.enabled:
            return

        # Handle CUDA synchronization for accurate GPU timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        if section_name not in self.active_sections:
            print(f"[PROFILER WARNING] Tried to end section '{section_name}' that was never started")
            return

        start_time = self.active_sections.pop(section_name)
        elapsed = end_time - start_time

        self.timings[section_name].append(elapsed)
        self.recent_timings[section_name].append(elapsed)
        self.call_counts[section_name] += 1

    def record_instant(self, section_name, elapsed_time):
        if not self.enabled:
            return

        self.timings[section_name].append(elapsed_time)
        self.recent_timings[section_name].append(elapsed_time)
        self.call_counts[section_name] += 1

    def get_stats(self, section_name):
        if section_name not in self.timings or len(self.timings[section_name]) == 0:
            return None

        times = self.timings[section_name]
        recent = list(self.recent_timings[section_name]) if section_name in self.recent_timings else times[-self.recent_window:]

        return {
            'count': self.call_counts[section_name],
            'total': sum(times),
            'mean': np.mean(times),
            'recent_mean': np.mean(recent) if recent else 0,
            'min': min(times),
            'max': max(times),
            'std': np.std(times),
        }

    def print_stats(self, top_n=15, title="Performance Profile"):
        if not self.enabled or not self.timings:
            return

        print()
        print("=" * 100)
        print(f"{title} (Step {self.step_count})")
        print("=" * 100)

        # Calculate stats for all sections
        stats = []
        for section_name in self.timings.keys():
            s = self.get_stats(section_name)
            if s:
                stats.append((section_name, s))

        # Sort by total time (descending)
        stats.sort(key=lambda x: x[1]['total'], reverse=True)

        # Print header
        print(f"{'Section':<45} {'Calls':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'Recent(ms)':>11} {'Min(ms)':>9} {'Max(ms)':>9}")
        print("-" * 100)

        # Print top N
        total_time = sum(s['total'] for _, s in stats)
        for i, (name, s) in enumerate(stats[:top_n]):
            pct = (s['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<45} {s['count']:>8} {s['total']:>10.3f} {s['mean']*1000:>10.3f} {s['recent_mean']*1000:>11.3f} {s['min']*1000:>9.3f} {s['max']*1000:>9.3f}  ({pct:>5.1f}%)")

        if len(stats) > top_n:
            print(f"... and {len(stats) - top_n} more sections")

        print("-" * 100)
        print(f"{'TOTAL':<45} {sum(s['count'] for _, s in stats):>8} {total_time:>10.3f}")
        print("=" * 100)
        print()

    def reset(self):
        self.timings.clear()
        self.call_counts.clear()
        self.active_sections.clear()
        self.recent_timings.clear()
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ProfiledSection:

    def __init__(self, profiler, section_name):
        self.profiler = profiler
        self.section_name = section_name

    def __enter__(self):
        self.profiler.start(self.section_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end(self.section_name)
        return False


# Global profiler instance (can be disabled via config)
_global_profiler = None

def get_profiler():
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled=False)  # Disabled for GUI
    return _global_profiler

def profile_section(section_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            if not profiler.enabled:
                return func(*args, **kwargs)

            profiler.start(section_name)
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.end(section_name)
            return result
        return wrapper
    return decorator

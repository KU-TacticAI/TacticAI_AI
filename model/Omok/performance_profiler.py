# Performance profiler copied from TicTacToe to Omok
# ...existing code...

import time
import functools
import logging
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
import json
import os
from datetime import datetime


class PerformanceProfiler:
    def __init__(self, log_file: Optional[str] = None, enable_console: bool = True):
        self.timers = {}
        self.stats = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'times': deque(maxlen=1000)
        })
        self.current_section = None
        self.section_stack = []
        self.logger = logging.getLogger('AlphaZeroProfiler')
        self.logger.setLevel(logging.INFO)
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def start_timer(self, name: str):
        self.timers[name] = time.time()
        if self.current_section:
            self.logger.info(f"[{self.current_section}] {name} 시작")
        else:
            self.logger.info(f"{name} 시작")

    def end_timer(self, name: str) -> float:
        if name not in self.timers:
            raise ValueError(f"타이머 '{name}'가 시작되지 않았습니다.")
        elapsed_time = time.time() - self.timers[name]
        stats = self.stats[name]
        stats['total_time'] += elapsed_time
        stats['count'] += 1
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], elapsed_time)
        stats['max_time'] = max(stats['max_time'], elapsed_time)
        stats['times'].append(elapsed_time)
        if self.current_section:
            self.logger.info(f"[{self.current_section}] {name} 완료: {elapsed_time:.4f}초")
        else:
            self.logger.info(f"{name} 완료: {elapsed_time:.4f}초")
        del self.timers[name]
        return elapsed_time

    def section(self, name: str):
        return SectionContext(self, name)

    def timer(self, name: str):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                self.start_timer(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(name)
            return wrapper
        return decorator

    def get_stats(self, name: Optional[str] = None) -> Dict:
        if name:
            return dict(self.stats[name])
        return {k: dict(v) for k, v in self.stats.items()}

    def print_summary(self):
        self.logger.info("=" * 60)
        self.logger.info("성능 프로파일링 요약")
        self.logger.info("=" * 60)
        sorted_stats = sorted(self.stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        for name, stats in sorted_stats:
            if stats['count'] > 0:
                self.logger.info(f"{name}:")
                self.logger.info(f"  총 시간: {stats['total_time']:.4f}초")
                self.logger.info(f"  호출 횟수: {stats['count']}")
                self.logger.info(f"  평균 시간: {stats['avg_time']:.4f}초")
                self.logger.info("")

    def save_report(self, filename: str):
        def convert_deque(d):
            return {k: (list(vv) if isinstance(vv, deque) else vv) for k, vv in d.items()}
        report = {'timestamp': datetime.now().isoformat(), 'stats': {k: convert_deque(v) for k, v in self.stats.items()}, 'summary': {}}
        total_time = sum(stats['total_time'] for stats in self.stats.values())
        report['summary']['total_time'] = total_time
        sorted_by_total = sorted(self.stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        report['summary']['top_time_consumers'] = [{'name': name, 'total_time': stats['total_time'], 'percentage': (stats['total_time'] / total_time * 100) if total_time > 0 else 0} for name, stats in sorted_by_total[:10]]
        sorted_by_count = sorted(self.stats.items(), key=lambda x: x[1]['count'], reverse=True)
        report['summary']['most_frequent'] = [{'name': name, 'count': stats['count']} for name, stats in sorted_by_count[:10]]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"성능 리포트가 {filename}에 저장되었습니다.")


class SectionContext:
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler.section_stack.append(self.name)
        self.profiler.current_section = self.name
        self.profiler.logger.info(f"섹션 시작: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.section_stack.pop()
        self.profiler.current_section = self.profiler.section_stack[-1] if self.profiler.section_stack else None
        self.profiler.logger.info(f"섹션 종료: {self.name}")


# 전역 프로파일러
_profiler = None


def get_profiler() -> PerformanceProfiler:
    global _profiler
    return _profiler


def profile_function(name: str):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = get_profiler()
            if profiler:
                profiler.start_timer(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profiler.end_timer(name)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_section(name: str):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = get_profiler()
            if profiler:
                with profiler.section(name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

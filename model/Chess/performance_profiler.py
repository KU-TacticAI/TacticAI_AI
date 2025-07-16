#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 프로파일링 모듈 (체스)

알파제로 학습 과정에서 각 단계별 시간을 측정하고 분석합니다.
"""

import time
import functools
import logging
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
import json
import os
from datetime import datetime


class PerformanceProfiler:
    """성능 프로파일링 클래스"""
    
    def __init__(self, log_file: Optional[str] = None, enable_console: bool = True):
        """
        프로파일러 초기화
        
        Args:
            log_file: 로그 파일 경로 (None이면 파일 로깅 비활성화)
            enable_console: 콘솔 출력 활성화 여부
        """
        self.timers = {}
        self.stats = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'times': deque(maxlen=1000)  # 최근 1000개 측정값 저장
        })
        self.current_section = None
        self.section_stack = []
        
        # 로깅 설정
        self.logger = logging.getLogger('AlphaZeroProfiler')
        self.logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def start_timer(self, name: str):
        """타이머 시작"""
        self.timers[name] = time.time()
        if self.current_section:
            self.logger.info(f"[{self.current_section}] {name} 시작")
        else:
            self.logger.info(f"{name} 시작")
    
    def end_timer(self, name: str) -> float:
        """타이머 종료 및 시간 반환"""
        if name not in self.timers:
            raise ValueError(f"타이머 '{name}'가 시작되지 않았습니다.")
        
        elapsed_time = time.time() - self.timers[name]
        
        # 통계 업데이트
        stats = self.stats[name]
        stats['total_time'] += elapsed_time
        stats['count'] += 1
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], elapsed_time)
        stats['max_time'] = max(stats['max_time'], elapsed_time)
        stats['times'].append(elapsed_time)
        
        # 로깅
        if self.current_section:
            self.logger.info(f"[{self.current_section}] {name} 완료: {elapsed_time:.4f}초")
        else:
            self.logger.info(f"{name} 완료: {elapsed_time:.4f}초")
        
        del self.timers[name]
        return elapsed_time
    
    def section(self, name: str):
        """섹션 컨텍스트 매니저"""
        return SectionContext(self, name)
    
    def timer(self, name: str):
        """타이머 데코레이터"""
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
        """통계 정보 반환"""
        if name:
            return dict(self.stats[name])
        return {k: dict(v) for k, v in self.stats.items()}
    
    def print_summary(self):
        """성능 요약 출력"""
        self.logger.info("=" * 60)
        self.logger.info("성능 프로파일링 요약")
        self.logger.info("=" * 60)
        
        # 총 시간 순으로 정렬
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        for name, stats in sorted_stats:
            if stats['count'] > 0:
                self.logger.info(f"{name}:")
                self.logger.info(f"  총 시간: {stats['total_time']:.4f}초")
                self.logger.info(f"  호출 횟수: {stats['count']}")
                self.logger.info(f"  평균 시간: {stats['avg_time']:.4f}초")
                self.logger.info(f"  최소 시간: {stats['min_time']:.4f}초")
                self.logger.info(f"  최대 시간: {stats['max_time']:.4f}초")
                self.logger.info("")
    
    def save_report(self, filename: str):
        """성능 리포트를 JSON 파일로 저장"""
        def convert_deque(d):
            return {k: (list(vv) if isinstance(vv, deque) else vv) for k, vv in d.items()}

        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': {k: convert_deque(v) for k, v in self.stats.items()},
            'summary': {}
        }
        
        # 요약 통계 계산
        total_time = sum(stats['total_time'] for stats in self.stats.values())
        report['summary']['total_time'] = total_time
        
        # 가장 시간이 오래 걸리는 작업들
        sorted_by_total = sorted(
            self.stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        report['summary']['top_time_consumers'] = [
            {'name': name, 'total_time': stats['total_time'], 'percentage': (stats['total_time'] / total_time * 100) if total_time > 0 else 0}
            for name, stats in sorted_by_total[:10]
        ]
        
        # 가장 자주 호출되는 작업들
        sorted_by_count = sorted(
            self.stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        report['summary']['most_frequent'] = [
            {'name': name, 'count': stats['count']}
            for name, stats in sorted_by_count[:10]
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"성능 리포트가 {filename}에 저장되었습니다.")


class SectionContext:
    """섹션 컨텍스트 매니저"""
    
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


# 전역 프로파일러 인스턴스
_profiler = None


def get_profiler() -> PerformanceProfiler:
    """전역 프로파일러 인스턴스 반환"""
    global _profiler
    return _profiler


def profile_function(name: str):
    """함수 프로파일링 데코레이터"""
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
    """섹션 프로파일링 데코레이터"""
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
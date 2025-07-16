#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
알파제로 틱텍토 학습 스크립트

이 스크립트는 알파제로 알고리즘을 사용하여 틱텍토 AI를 학습시킵니다.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import signal
import multiprocessing as mp
import atexit

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.TicTacToe.alphazero import AlphaZero

# 전역 변수로 AlphaZero 인스턴스 저장
alphazero_instance = None

def cleanup_processes():
    """프로세스 정리 함수"""
    global alphazero_instance
    if alphazero_instance and hasattr(alphazero_instance, 'current_pool') and alphazero_instance.current_pool:
        print("\n프로세스 풀을 정리합니다...")
        alphazero_instance.current_pool.terminate()
        alphazero_instance.current_pool.join()
        alphazero_instance.current_pool = None

def signal_handler(signum, frame):
    """시그널 핸들러 - Ctrl+C 등으로 종료 시 모든 프로세스 정리"""
    print(f"\n시그널 {signum}을 받았습니다. 모든 프로세스를 정리하고 종료합니다...")
    
    # 전역 정리 함수 호출
    cleanup_processes()
    
    # 모든 자식 프로세스 종료
    for process in mp.active_children():
        print(f"프로세스 {process.pid}를 종료합니다...")
        process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            print(f"프로세스 {process.pid}를 강제 종료합니다...")
            process.kill()
            process.join(timeout=2)
    
    print("프로그램을 종료합니다.")
    os._exit(0)  # 강제 종료

def main():
    """메인 함수"""
    global alphazero_instance
    
    # 종료 시 정리 함수 등록
    atexit.register(cleanup_processes)
    
    # 시그널 핸들러 설정
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='알파제로 틱텍토 학습')
    
    # 학습 파라미터
    parser.add_argument('--iterations', type=int, default=50,
                       help='전체 반복 횟수 (기본값: 50)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='각 반복당 자기대전 게임 수 (기본값: 50)')
    parser.add_argument('--mcts_sims', type=int, default=400,
                       help='MCTS 시뮬레이션 수 (기본값: 400)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='신경망 학습 에포크 수 (기본값: 5)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='배치 크기 (기본값: 256)')
    parser.add_argument('--mcts_batch_size', type=int, default=32,
                       help='MCTS 배치 크기 (기본값: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='학습률 (기본값: 0.002)')
    
    # MCTS 파라미터
    parser.add_argument('--c_puct', type=float, default=1.0,
                       help='MCTS UCB 상수 (기본값: 1.0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='온도 파라미터 (기본값: 1.0)')
    
    # 모델 저장/로드
    parser.add_argument('--model_dir', type=str, default=None,
                       help='모델 저장 디렉토리 (기본값: train.py와 같은 폴더의 models)')
    parser.add_argument('--load_iteration', type=int, default=None,
                       help='로드할 반복 번호 (기본값: None)')
    parser.add_argument('--load_checkpoint', type=int, default=None,
                       help='로드할 체크포인트 반복 번호 (기본값: None)')
    parser.add_argument('--load_optimizer', action='store_true', default=True,
                       help='옵티마이저 상태 로드 (기본값: True)')
    parser.add_argument('--load_buffer', action='store_true', default=True,
                       help='리플레이 버퍼 로드 (기본값: True)')
    parser.add_argument('--load_metadata', action='store_true', default=True,
                       help='메타데이터 로드 (기본값: True)')
    parser.add_argument('--no_resume_training', action='store_true',
                       help='학습 재개하지 않음 (기본값: False)')
    
    # 디바이스 설정
    parser.add_argument('--device', type=str, default=None,
                       help='사용할 디바이스 (cuda/cpu, 기본값: 자동)')
    parser.add_argument('--cpu_only', action='store_true',
                       help='CPU만 사용 (기본값: False)')
    
    # 평가 설정
    parser.add_argument('--evaluate_only', action='store_true',
                       help='학습 없이 평가만 수행 (기본값: False)')
    parser.add_argument('--eval_games', type=int, default=100,
                       help='평가 게임 수 (기본값: 100)')
    
    # 사람과 대전
    parser.add_argument('--play_human', action='store_true',
                       help='사람과 대전 (기본값: False)')
    
    # 프로파일링 설정
    parser.add_argument('--enable_profiling', action='store_true', default=True,
                       help='성능 프로파일링 활성화 (기본값: True)')
    parser.add_argument('--disable_profiling', action='store_true',
                       help='성능 프로파일링 비활성화')
    
    args = parser.parse_args()
    
    # 모델 디렉토리 경로를 train.py와 같은 위치 기준으로 고정 (절대 경로 사용)
    if args.model_dir is None:
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / 'models'
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            script_dir = Path(__file__).parent.resolve()
            model_dir = script_dir / model_dir
    
    # 절대 경로로 변환하여 확실하게 설정
    model_dir = model_dir.resolve()
    print(f"모델 저장 경로: {model_dir}")
    
    # 디바이스 설정
    if args.cpu_only:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 디바이스: {device}")
    
    # 프로파일링 설정
    enable_profiling = args.enable_profiling and not args.disable_profiling
    
    # 알파제로 초기화
    alphazero_instance = AlphaZero(
        model_dir=str(model_dir),
        num_iterations=args.iterations,
        num_episodes=args.episodes,
        num_mcts_sims=args.mcts_sims,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        c_puct=args.c_puct,
        temperature=args.temperature,
        device=device,
        mcts_batch_size=args.mcts_batch_size,
        enable_profiling=enable_profiling
    )
    
    # 모델 로드 (지정된 경우)
    if args.load_checkpoint is not None:
        # 체크포인트 로드
        alphazero_instance.load_checkpoint(args.load_checkpoint)
    elif args.load_iteration is not None:
        # 일반 모델 로드
        alphazero_instance.load_model(
            args.load_iteration,
            load_optimizer=args.load_optimizer,
            load_buffer=args.load_buffer,
            load_metadata=args.load_metadata,
            resume_training=not args.no_resume_training
        )
    
    # 평가만 수행
    if args.evaluate_only:
        print("모델 평가 중...")
        win_rate = alphazero_instance.evaluate_model(args.eval_games)
        print(f"최종 승률: {win_rate:.2%}")
        return
    
    # 사람과 대전
    if args.play_human:
        alphazero_instance.play_against_human()
        return
    
    # 학습 시작
    print("알파제로 틱텍토 학습을 시작합니다!")
    print(f"설정:")
    print(f"  - 반복 횟수: {args.iterations}")
    print(f"  - 각 반복당 게임 수: {args.episodes}")
    print(f"  - MCTS 시뮬레이션 수: {args.mcts_sims}")
    print(f"  - 학습 에포크 수: {args.epochs}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - MCTS 배치 크기: {args.mcts_batch_size}")
    print(f"  - 학습률: {args.learning_rate}")
    print(f"  - MCTS UCB 상수: {args.c_puct}")
    print(f"  - 온도: {args.temperature}")
    print(f"  - 모델 저장 디렉토리: {model_dir}")
    print(f"  - 성능 프로파일링: {'활성화' if enable_profiling else '비활성화'}")
    print("\nCtrl+C를 눌러 언제든지 안전하게 종료할 수 있습니다.")
    
    try:
        # 멀티프로세싱 시작 방식 설정 (시그널 핸들링을 위해)
        if mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn', force=True)
        
        alphazero_instance.train()
        print("학습이 성공적으로 완료되었습니다!")
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")
        cleanup_processes()
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")
        cleanup_processes()
        raise
    finally:
        cleanup_processes()


if __name__ == "__main__":
    main() 
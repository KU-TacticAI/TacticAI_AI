#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omok AlphaZero 학습 스크립트 (TicTacToe 스타일로 맞춤)
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

from model.Omok.alphazero import AlphaZeroOmok

alphazero_instance = None

def cleanup_processes():
    global alphazero_instance
    if alphazero_instance and hasattr(alphazero_instance, 'current_pool') and alphazero_instance.current_pool:
        print("\n프로세스 풀을 정리합니다...")
        try:
            alphazero_instance.current_pool.terminate()
            alphazero_instance.current_pool.join()
        except Exception:
            pass
        alphazero_instance.current_pool = None


def signal_handler(signum, frame):
    print(f"\n시그널 {signum}을 받았습니다. 모든 프로세스를 정리하고 종료합니다...")
    cleanup_processes()
    for process in mp.active_children():
        try:
            process.terminate()
            process.join(timeout=3)
        except Exception:
            pass
    os._exit(0)


def main():
    global alphazero_instance
    atexit.register(cleanup_processes)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description='알파제로 Omok 학습')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--mcts_sims', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mcts_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--board_size', type=int, default=15)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--cpu_only', action='store_true')
    parser.add_argument('--play_human', action='store_true')
    args = parser.parse_args()

    if args.model_dir is None:
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / 'models'
    else:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            script_dir = Path(__file__).parent.resolve()
            model_dir = script_dir / model_dir
    model_dir = model_dir.resolve()
    print(f"모델 저장 경로: {model_dir}")

    if args.cpu_only:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    alphazero_instance = AlphaZeroOmok(
        model_dir=str(model_dir),
        board_size=args.board_size,
        num_iterations=args.iterations,
        num_episodes=args.episodes,
        num_mcts_sims=args.mcts_sims,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mcts_batch_size=args.mcts_batch_size,
    )

    if args.play_human:
        alphazero_instance.play_against_human()
        return

    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        alphazero_instance.train()
    except KeyboardInterrupt:
        print('\n학습이 중단되었습니다.')
        cleanup_processes()
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        cleanup_processes()
        raise
    finally:
        cleanup_processes()


if __name__ == '__main__':
    main()

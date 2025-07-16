"""
10길이 배열을 직접 처리하는 JIT 모델 변환 스크립트 (틱택토)

입력: 10길이 배열 (턴 + 보드)
출력: 9길이 정책 확률
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.TicTacToe.neural_network import TicTacToeNeuralNetwork

class Direct10JITWrapper(nn.Module):
    """10길이 배열을 직접 처리하는 JIT 래퍼 클래스 (틱택토)"""
    def __init__(self, original_model: TicTacToeNeuralNetwork):
        super().__init__()
        self.original_model = original_model
    def forward(self, array_10: torch.Tensor) -> torch.Tensor:
        """
        순전파 - 10길이 배열을 직접 처리
        Args:
            array_10: 10길이 배열 (batch_size, 10)
                - 인덱스 0: 턴 정보 (1: X, -1: O)
                - 인덱스 1-9: 보드 상황 (X: 1, O: -1, 빈칸: 0)
        Returns:
            policy: 정책 확률 (batch_size, 9)
        """
        batch_size = array_10.shape[0]
        encoded_states = []
        for i in range(batch_size):
            current_player = array_10[i, 0]
            board_1d = array_10[i, 1:10]
            board_2d = board_1d.reshape(3, 3)
            encoded = torch.zeros(3, 3, 3, device=array_10.device, dtype=array_10.dtype)
            encoded[0] = (board_2d == current_player).float()
            encoded[1] = (board_2d == -current_player).float()
            encoded[2] = (current_player == 1).float()
            encoded_states.append(encoded)
        encoded_batch = torch.stack(encoded_states)
        policy, _ = self.original_model(encoded_batch)
        return policy

def convert_model_to_jit(model_path: str, output_path: str, device: str = None) -> str:
    """
    모델을 JIT로 변환 (틱택토)
    Args:
        model_path: 원본 모델 파일 경로
        output_path: 출력 JIT 모델 파일 경로
        device: 사용할 디바이스
    Returns:
        output_path: 생성된 JIT 모델 파일 경로
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"원본 모델 로딩 중: {model_path}")
    original_model = TicTacToeNeuralNetwork.load_model(model_path, device)
    original_model.eval()
    jit_wrapper = Direct10JITWrapper(original_model)
    jit_wrapper.eval()
    test_input = torch.randn(1, 10, device=device)
    print("JIT 모델 변환 중...")
    with torch.no_grad():
        jit_model = torch.jit.trace(jit_wrapper, test_input)
    print(f"JIT 모델 저장 중: {output_path}")
    jit_model.save(output_path)
    print("JIT 모델 변환 완료!")
    return output_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="10길이 배열을 직접 처리하는 JIT 모델 변환 (틱택토)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="원본 모델 파일 경로 (.pth)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="출력 JIT 모델 파일 경로 (.pt)")
    parser.add_argument("--device", type=str, default=None,
                       help="사용할 디바이스 (cuda/cpu)")
    args = parser.parse_args()
    convert_model_to_jit(args.model_path, args.output_path, args.device)

if __name__ == "__main__":
    main() 
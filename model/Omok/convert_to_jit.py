"""
Omok 모델을 JIT로 변환하는 스크립트

입력: 1 + board_size*board_size 길이 배열 (턴 정보 + 보드 flatten)
출력: 정책 logits/probs (board_size*board_size)

TicTacToe의 convert_to_jit.py 스타일을 따릅니다.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.Omok.neural_network import OmokNeuralNetwork


class DirectFlattenJITWrapper(nn.Module):
    """플랫한 입력(턴 + 1D 보드)을 받아 내부 모델에 전달하는 JIT 래퍼"""

    def __init__(self, original_model: OmokNeuralNetwork):
        super().__init__()
        self.original_model = original_model
        self.board_size = original_model.board_size

    def forward(self, array_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            array_in: (batch_size, 1 + board_size*board_size)
                인덱스 0: 턴 정보 (1 또는 -1)
                인덱스 1..: 보드 1D (1: 흑, -1: 백, 0: 빈칸)
        Returns:
            policy: (batch_size, board_size*board_size) - 로그 확률(log_softmax) 형태
        """
        batch_size = array_in.shape[0]
        bs = self.board_size
        # 기대되는 길이 검사 (JIT에서는 런타임에 고정되지 않을 수 있음)
        # 배열을 (batch, 3, H, W)로 인코딩
        encoded = torch.zeros((batch_size, 3, bs, bs), dtype=array_in.dtype, device=array_in.device)
        for i in range(batch_size):
            current_player = array_in[i, 0]
            board_flat = array_in[i, 1:1 + bs * bs]
            board_2d = board_flat.view(bs, bs)
            # 채널 0: 현재 플레이어의 돌
            encoded[i, 0] = (board_2d == current_player).to(dtype=array_in.dtype)
            # 채널 1: 상대방 돌
            encoded[i, 1] = (board_2d == (-current_player)).to(dtype=array_in.dtype)
            # 채널 2: 현재 플레이어 표시
            encoded[i, 2].fill_(1.0 if current_player == 1 else 0.0)

        policy_logits, value = self.original_model(encoded)
        # 정책(로그확률)을 그대로 반환 (외부에서 exp 사용 가능)
        return policy_logits


def convert_model_to_jit(model_path: str, output_path: str, device: str = None) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"원본 모델 로딩: {model_path}")
    original_model = OmokNeuralNetwork.load_model(model_path, device)
    original_model.eval()

    wrapper = DirectFlattenJITWrapper(original_model)
    wrapper.eval()

    bs = original_model.board_size
    example_input = torch.zeros((1, 1 + bs * bs), dtype=torch.float32, device=device)
    example_input[0, 0] = 1.0  # 흑 차례 예시

    print("JIT 변환을 수행합니다...")
    with torch.no_grad():
        jit_model = torch.jit.trace(wrapper, example_input)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"JIT 모델 저장: {output_path}")
    jit_model.save(output_path)
    print("변환 완료")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Omok 모델을 JIT로 변환')
    parser.add_argument('--model_path', type=str, required=True, help='원본 .pth 모델 파일 경로')
    parser.add_argument('--output_path', type=str, required=True, help='출력 .pt JIT 파일 경로')
    parser.add_argument('--device', type=str, default=None, help='cuda 또는 cpu')
    args = parser.parse_args()

    convert_model_to_jit(args.model_path, args.output_path, args.device)


if __name__ == '__main__':
    main()

"""
73길이 배열을 직접 처리하는 JIT 모델 변환 스크립트 (체스)

입력: 73길이 배열 (턴 + 보드)
출력: 4672길이 정책 확률
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

from model.Chess.neural_network import ChessNeuralNetwork

class Direct70JITWrapper(nn.Module):
    """70길이 배열(턴+보드+캐슬링+앙파상)을 18채널로 인코딩하는 JIT 래퍼 (체스)"""
    def __init__(self, original_model: ChessNeuralNetwork):
        super().__init__()
        self.original_model = original_model
    def forward(self, array_70: torch.Tensor) -> torch.Tensor:
        """
        순전파 - 70길이 배열을 18채널로 인코딩
        Args:
            array_70: 70길이 배열 (batch_size, 70)
                - 0: 턴 정보 (1: 백, 0: 흑)
                - 1~64: 8x8 체스 보드 (row-major, 0=빈칸, 1~6=백, -1~-6=흑)
                - 65~68: 캐슬링 권리 (백킹, 백퀸, 흑킹, 흑퀸, 1: 가능, 0: 불가)
                - 69: 앙파상 타겟 (0~63: 칸 인덱스, -1: 없음)
        Returns:
            policy: 정책 확률 (batch_size, 4672)
        """
        batch_size = array_70.shape[0]
        encoded_states = []
        for i in range(batch_size):
            board_1d = array_70[i, 1:65]
            board_2d = board_1d.reshape(8, 8)
            encoded = torch.zeros(18, 8, 8, device=array_70.device, dtype=array_70.dtype)
            # 12채널: 백/흑 기물
            for r in range(8):
                for c in range(8):
                    v = board_2d[r, c].item()
                    if v == 1:   encoded[0, r, c] = 1  # 백 폰
                    elif v == 2: encoded[1, r, c] = 1  # 백 나이트
                    elif v == 3: encoded[2, r, c] = 1  # 백 비숍
                    elif v == 4: encoded[3, r, c] = 1  # 백 룩
                    elif v == 5: encoded[4, r, c] = 1  # 백 퀸
                    elif v == 6: encoded[5, r, c] = 1  # 백 킹
                    elif v == -1:  encoded[6, r, c] = 1  # 흑 폰
                    elif v == -2:  encoded[7, r, c] = 1  # 흑 나이트
                    elif v == -3:  encoded[8, r, c] = 1  # 흑 비숍
                    elif v == -4:  encoded[9, r, c] = 1  # 흑 룩
                    elif v == -5:  encoded[10, r, c] = 1 # 흑 퀸
                    elif v == -6:  encoded[11, r, c] = 1 # 흑 킹
            # 턴 정보 (채널 12)
            turn = array_70[i, 0]
            encoded[12, :, :] = 1 if turn == 1 else 0
            # 캐슬링 권리 (채널 13~16)
            for j in range(4):
                encoded[13+j, :, :] = array_70[i, 65+j]
            # 앙파상 (채널 17)
            ep_idx = int(array_70[i, 69].item())
            if 0 <= ep_idx < 64:
                r, c = divmod(ep_idx, 8)
                encoded[17, r, c] = 1
            encoded_states.append(encoded)
        encoded_batch = torch.stack(encoded_states)
        policy, value = self.original_model(encoded_batch)
        return policy

def convert_model_to_jit(model_path: str, output_path: str, device: str = None) -> str:
    """
    모델을 JIT로 변환 (체스)
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
    original_model = ChessNeuralNetwork.load_model(model_path, device)
    original_model.eval()
    jit_wrapper = Direct70JITWrapper(original_model)
    jit_wrapper.eval()
    test_input = torch.zeros(1, 70, device=device)  # 예시 입력
    print("JIT 모델 변환 중...")
    with torch.no_grad():
        jit_model = torch.jit.trace(jit_wrapper, test_input)
    print(f"JIT 모델 저장 중: {output_path}")
    jit_model.save(output_path)
    print("JIT 모델 변환 완료!")
    return output_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="73길이 배열을 직접 처리하는 JIT 모델 변환 (체스)")
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
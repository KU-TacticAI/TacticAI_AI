#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
65길이 배열을 직접 처리하는 JIT 모델 변환 스크립트

입력: 65길이 배열 (턴 + 보드)
출력: 65길이 정책 확률
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

from model.Othello.neural_network import OthelloNeuralNetwork


class Direct65JITWrapper(nn.Module):
    """65길이 배열을 직접 처리하는 JIT 래퍼 클래스"""
    
    def __init__(self, original_model: OthelloNeuralNetwork):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, array_65: torch.Tensor) -> torch.Tensor:
        """
        순전파 - 65길이 배열을 직접 처리
        
        Args:
            array_65: 65길이 배열 (batch_size, 65)
                - 인덱스 0: 턴 정보 (1: 흑돌, -1: 백돌)
                - 인덱스 1-64: 보드 상황 (검은돌: 1, 흰돌: -1, 빈칸: 0)
            
        Returns:
            policy: 정책 확률 (batch_size, 65)
        """
        batch_size = array_65.shape[0]
        
        # 3채널 인코딩으로 변환
        encoded_states = []
        for i in range(batch_size):
            current_player = array_65[i, 0]
            board_1d = array_65[i, 1:65]
            board_2d = board_1d.reshape(8, 8)
            
            encoded = torch.zeros(3, 8, 8, device=array_65.device, dtype=array_65.dtype)
            encoded[0] = (board_2d == current_player).float()
            encoded[1] = (board_2d == -current_player).float()
            encoded[2] = (current_player == 1).float()
            
            encoded_states.append(encoded)
        
        encoded_batch = torch.stack(encoded_states)
        policy, _ = self.original_model(encoded_batch)
        
        return policy


class Direct65JITInference:
    """65길이 배열을 직접 처리하는 JIT 추론 클래스"""
    
    def __init__(self, jit_model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = torch.jit.load(jit_model_path, map_location=self.device)
        self.model.eval()
    
    def predict(self, array_65: np.ndarray) -> np.ndarray:
        """
        단일 상태에 대한 추론
        
        Args:
            array_65: 65길이 배열
                - 인덱스 0: 턴 정보 (1: 흑돌, -1: 백돌)
                - 인덱스 1-64: 보드 상황 (검은돌: 1, 흰돌: -1, 빈칸: 0)
            
        Returns:
            policy: 정책 확률 (65,)
        """
        if array_65.shape != (65,):
            raise ValueError(f"입력 배열은 (65,) 형태여야 합니다. 현재: {array_65.shape}")
        
        array_tensor = torch.FloatTensor(array_65).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy = self.model(array_tensor)
        
        return policy.cpu().numpy()[0]
    
    def predict_batch(self, arrays_65: np.ndarray) -> np.ndarray:
        """
        배치 상태에 대한 추론
        
        Args:
            arrays_65: 65길이 배열 배치 (batch_size, 65)
            
        Returns:
            policies: 정책 확률 배치 (batch_size, 65)
        """
        if len(arrays_65.shape) != 2 or arrays_65.shape[1] != 65:
            raise ValueError(f"입력 배열은 (batch_size, 65) 형태여야 합니다. 현재: {arrays_65.shape}")
        
        array_tensor = torch.FloatTensor(arrays_65).to(self.device)
        
        with torch.no_grad():
            policies = self.model(array_tensor)
        
        return policies.cpu().numpy()


def convert_model_to_jit(model_path: str, output_path: str, device: str = None) -> str:
    """
    모델을 JIT로 변환
    
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
    original_model = OthelloNeuralNetwork.load_model(model_path, device)
    original_model.eval()
    
    jit_wrapper = Direct65JITWrapper(original_model)
    jit_wrapper.eval()
    
    test_input = torch.randn(1, 65, device=device)
    
    print("JIT 모델 변환 중...")
    with torch.no_grad():
        jit_model = torch.jit.trace(jit_wrapper, test_input)
    
    print(f"JIT 모델 저장 중: {output_path}")
    jit_model.save(output_path)
    
    print("JIT 모델 변환 완료!")
    return output_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="65길이 배열을 직접 처리하는 JIT 모델 변환")
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
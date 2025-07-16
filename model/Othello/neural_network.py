"""
알파제로용 오셀로 신경망 모델

정책 네트워크와 가치 네트워크를 포함한 듀얼 네트워크 구조를 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os


class ResidualBlock(nn.Module):
    """ResNet 스타일의 잔차 블록"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class OthelloNeuralNetwork(nn.Module):
    """알파제로용 오셀로 신경망 모델"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_residual_blocks: int = 10,
                 num_filters: int = 256,
                 num_actions: int = 65):  # 64 + 1(pass)
        """
        신경망 초기화
        
        Args:
            input_channels: 입력 채널 수 (기본값: 3)
            num_residual_blocks: 잔차 블록 수 (기본값: 10)
            num_filters: 필터 수 (기본값: 256)
            num_actions: 액션 수 (기본값: 65 = 8x8+1)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.num_actions = num_actions
        
        # 초기 컨볼루션 레이어
        self.conv_input = nn.Conv2d(input_channels, num_filters, 
                                   kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # 정책 헤드 (Policy Head)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_actions)  # 65
        
        # 가치 헤드 (Value Head)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch_size, channels, height, width)
            
        Returns:
            policy: 정책 출력 (batch_size, num_actions)
            value: 가치 출력 (batch_size, 1)
        """
        # 입력 처리
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 잔차 블록들
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # 정책 헤드
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, state: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, float]:
        """
        단일 상태에 대한 예측
        
        Args:
            state: 게임 상태 (3, 8, 8)
            device: 사용할 디바이스
            
        Returns:
            policy: 정책 확률 (65,)
            value: 가치 예측
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            # 배치 차원 추가
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy_logits, value = self.forward(state_tensor)
            
            # 정책을 확률로 변환
            policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0, 0]
            
            return policy_probs, value_scalar
    
    def predict_batch(self, states: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        배치 상태에 대한 예측
        
        Args:
            states: 게임 상태 배치 (batch_size, 3, 8, 8)
            device: 사용할 디바이스
            
        Returns:
            policies: 정책 확률 배치 (batch_size, 65)
            values: 가치 예측 배치 (batch_size,)
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
            policy_logits, values = self.forward(states_tensor)
            
            # 정책을 확률로 변환
            policy_probs = torch.exp(policy_logits).cpu().numpy()
            values_scalar = values.cpu().numpy().flatten()
            
            return policy_probs, values_scalar
    
    def save_model(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'num_residual_blocks': self.num_residual_blocks,
            'num_filters': self.num_filters,
            'num_actions': self.num_actions
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'OthelloNeuralNetwork':
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_channels=checkpoint['input_channels'],
            num_residual_blocks=checkpoint['num_residual_blocks'],
            num_filters=checkpoint['num_filters'],
            num_actions=checkpoint['num_actions']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model


class PolicyValueLoss(nn.Module):
    """정책과 가치 손실 함수"""
    
    def __init__(self, value_weight: float = 1.0):
        """
        손실 함수 초기화
        
        Args:
            value_weight: 가치 손실의 가중치
        """
        super().__init__()
        self.value_weight = value_weight
    
    def forward(self, 
                policy_pred: torch.Tensor, 
                value_pred: torch.Tensor,
                policy_target: torch.Tensor, 
                value_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        손실 계산
        
        Args:
            policy_pred: 예측된 정책 (log_softmax)
            value_pred: 예측된 가치
            policy_target: 목표 정책
            value_target: 목표 가치
            
        Returns:
            total_loss: 총 손실
            policy_loss: 정책 손실
            value_loss: 가치 손실
        """
        # 정책 손실 (Cross Entropy)
        policy_loss = F.kl_div(policy_pred, policy_target, reduction='batchmean')
        
        # 가치 손실 (MSE)
        value_loss = F.mse_loss(value_pred, value_target)
        
        # 총 손실
        total_loss = policy_loss + self.value_weight * value_loss
        
        return total_loss, policy_loss, value_loss 
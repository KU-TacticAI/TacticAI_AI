"""
AlphaZero용 틱택토 신경망

정책 네트워크와 가치 네트워크를 포함한 PyTorch 모델입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import os

class ResidualBlock(nn.Module):
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

class TicTacToeNeuralNetwork(nn.Module):
    def __init__(self, input_channels: int = 3, num_residual_blocks: int = 3, num_filters: int = 64, num_actions: int = 9):
        super().__init__()
        self.input_channels = input_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.num_actions = num_actions
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 3 * 3, num_actions)  # 9
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 3 * 3, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self._initialize_weights()
    def _initialize_weights(self):
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
        x = F.relu(self.bn_input(self.conv_input(x)))
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        # 정책 헤드
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(-1, 2 * 3 * 3)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(-1, 1 * 3 * 3)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    def predict(self, state: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, float]:
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)  # (1, 3, 3, 3) -> (1, 3, 3, 3)
            policy_logits, value = self.forward(state_tensor)
            policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0, 0]
            return policy_probs, value_scalar
    def predict_batch(self, states: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)  # (batch, 3, 3, 3)
            policy_logits, values = self.forward(states_tensor)
            policy_probs = torch.exp(policy_logits).cpu().numpy()
            values_scalar = values.cpu().numpy().flatten()
            return policy_probs, values_scalar
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_channels': self.input_channels,
            'num_residual_blocks': self.num_residual_blocks,
            'num_filters': self.num_filters,
            'num_actions': self.num_actions
        }, filepath)
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'TicTacToeNeuralNetwork':
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_channels=checkpoint['input_channels'],
            num_residual_blocks=checkpoint['num_residual_blocks'],
            num_filters=checkpoint['num_filters'],
            num_actions=checkpoint['num_actions']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if device is not None:
            model.to(device)
        return model

class PolicyValueLoss(nn.Module):
    def __init__(self, value_weight: float = 1.0):
        super().__init__()
        self.value_weight = value_weight
    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor, policy_target: torch.Tensor, value_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_loss = F.kl_div(policy_pred, policy_target, reduction='batchmean')
        value_loss = F.mse_loss(value_pred.view(-1), value_target.view(-1))
        total_loss = policy_loss + self.value_weight * value_loss
        return total_loss, policy_loss, value_loss 
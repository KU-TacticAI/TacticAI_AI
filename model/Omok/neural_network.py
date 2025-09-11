"""
Omok(오목)용 알파제로 신경망 모델

Othello/TicTacToe 스타일을 따라 만든 가변 보드 크기 대응 ResNet 기반 정책+가치 신경망입니다.
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


class OmokNeuralNetwork(nn.Module):
    """Omok용 ResNet 스타일 정책+가치 신경망

    인코딩은 (batch, 3, H, W)
    정책 헤드는 보드 크기에 따라 출력 차원이 결정됩니다.
    """

    def __init__(self, input_channels: int = 3, num_residual_blocks: int = 8, num_filters: int = 128, board_size: int = 15):
        super().__init__()
        self.input_channels = input_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        self.num_actions = board_size * board_size

        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)

        # value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

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
        for block in self.residual_blocks:
            x = block(x)

        # policy
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # value
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, state: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, float]:
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy_logits, value = self.forward(state_tensor)
            policy_probs = torch.exp(policy_logits).cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0, 0]
            return policy_probs, value_scalar

    def predict_batch(self, states: np.ndarray, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(device)
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
            'board_size': self.board_size
        }, filepath)

    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'OmokNeuralNetwork':
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_channels=checkpoint.get('input_channels', 3),
            num_residual_blocks=checkpoint.get('num_residual_blocks', 8),
            num_filters=checkpoint.get('num_filters', 128),
            board_size=checkpoint.get('board_size', 15)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if device is not None:
            model = model.to(device)
        return model


class PolicyValueLoss(nn.Module):
    def __init__(self, value_weight: float = 1.0):
        super().__init__()
        self.value_weight = value_weight

    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor, policy_target: torch.Tensor, value_target: torch.Tensor):
        policy_loss = F.kl_div(policy_pred, policy_target, reduction='batchmean')
        value_loss = F.mse_loss(value_pred, value_target)
        total_loss = policy_loss + self.value_weight * value_loss
        return total_loss, policy_loss, value_loss

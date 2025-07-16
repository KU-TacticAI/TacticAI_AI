import torch
import numpy as np
import time

# JIT 모델 로드
jit_model = torch.jit.load("models/othello_jit.pt", map_location="cuda")
jit_model.eval()

# 입력 준비 (예: 단일 상태)
array_65 = np.zeros(65, dtype=np.float32)
array_65[0] = 1  # 턴 정보
# array_65[1:65] = ...  # 보드 정보

input_tensor = torch.from_numpy(array_65).unsqueeze(0).to("cuda")  # (1, 65)

with torch.no_grad():
    policy = jit_model(input_tensor)  # (1, 65)
    policy = policy.cpu().numpy()[0]  # (65,)

print(policy)
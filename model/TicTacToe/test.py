import torch
import numpy as np
import time

# JIT 모델 로드
jit_model = torch.jit.load("models/tictactoe_jit.pt", map_location="cuda")
jit_model.eval()

# 입력 준비 (예: 단일 상태)
array_10 = np.zeros(10, dtype=np.float32)
array_10[0] = 1  # 턴 정보 (X의 턴)
# array_10[1:10] = ...  # 보드 정보 (X=1, O=-1, 빈칸=0)

input_tensor = torch.from_numpy(array_10).unsqueeze(0).to("cuda")  # (1, 10)

with torch.no_grad():
    policy = jit_model(input_tensor)  # (1, 9)
    policy = policy.cpu().numpy()[0]  # (9,)

print(policy)

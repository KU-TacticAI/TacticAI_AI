import torch
import numpy as np
import time

# JIT 모델 로드
jit_model = torch.jit.load("models/chess_jit.pt", map_location="cuda")
jit_model.eval()

# 입력 준비 (예: 단일 상태)
# JIT 모델은 70길이 배열을 입력으로 받음:
# [0]: 턴 정보 (1=백, 0=흑)
# [1~64]: 8x8 보드 (row-major, 0=빈칸, 1~6=백, -1~-6=흑)
# [65~68]: 캐슬링 권리 (백킹, 백퀸, 흑킹, 흑퀸, 1=가능, 0=불가)
# [69]: 앙파상 타겟 (0~63: 칸 인덱스, -1: 없음)
array_70 = np.zeros(70, dtype=np.float32)
array_70[0] = 1  # 턴 정보 (백의 턴)
# array_70[1:65] = ...  # 보드 정보 (각 말 종류별 값)
# array_70[65:69] = ... # 캐슬링 권리
# array_70[69] = -1    # 앙파상 타겟 없을 때

input_tensor = torch.from_numpy(array_70).unsqueeze(0).to("cuda")  # (1, 70)

with torch.no_grad():
    policy = jit_model(input_tensor)  # (1, 4672)
    policy = policy.cpu().numpy()[0]  # (4672,)

print(policy)
print(len(policy))
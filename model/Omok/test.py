"""
간단한 Omok 테스트 스크립트 (TicTacToe/test.py 스타일)
"""
import torch
import numpy as np
import time
from pathlib import Path

# JIT 모델이 있을 경우 로드 예시
model_path = Path(__file__).parent / 'models' / 'omok_jit.pt'
if model_path.exists():
    jit_model = torch.jit.load(str(model_path), map_location='cuda')
    jit_model.eval()

    # 입력 준비 (예: 단일 상태)
    bs = 15
    array_len = 1 + bs * bs
    array = np.zeros(array_len, dtype=np.float32)
    array[0] = 1  # 턴 정보 (흑 차례)
    # array[1:] = ...  # 보드 정보 (흑=1, 백=-1, 빈칸=0)

    input_tensor = torch.from_numpy(array).unsqueeze(0).to("cuda")  # (1, array_len)

    with torch.no_grad():
        policy = jit_model(input_tensor)  # (1, bs*bs)
        policy = policy.cpu().numpy()[0]  # (bs*bs,)

    print(policy)
    print(len(policy))  # 225
else:
    print('JIT 모델 파일이 없습니다:', model_path)

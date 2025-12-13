#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 매니저

AI 모델의 로드, 캐싱, 추론을 담당하는 매니저입니다.
PyTorch 모델을 관리하고 추론 요청을 처리합니다.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import requests
from urllib.parse import urlparse
import time

from config import Config
from message_schemas import (
  InferenceRequest, InferenceResponse,
  ModelLoadRequest, ModelLoadResponse, GameType
)


class AIModel:
  """AI 모델 래퍼 클래스"""

  def __init__(self, model_id: str, model_path: str, game_type: GameType):
    """
        AI 모델 초기화

        Args:
            model_id: 모델 ID
            model_path: 모델 파일 경로
            game_type: 게임 타입
        """
    self.model_id = model_id
    self.model_path = model_path
    self.game_type = game_type
    self.model = None
    self.device = torch.device(Config.DEVICE)
    self.logger = logging.getLogger(__name__)

  def load(self) -> bool:
    """
        모델 로드

        Returns:
            bool: 로드 성공 여부
        """
    try:
      self.model = torch.jit.load(self.model_path, map_location=self.device)
      self.model.eval()
      self.model.to(self.device)
      self.logger.info(
        f"Model {self.model_id} loaded to device: {self.device} (is_cuda={self.device.type == 'cuda'})")
      return True
    except Exception as e:
      self.logger.error(f"Failed to load model {self.model_id}: {e}")
      return False

  def unload(self):
    """모델 언로드"""
    try:
      del self.model
      torch.cuda.empty_cache()
      self.logger.info(f"Model {self.model_id} unloaded from memory.")
    except Exception as e:
      self.logger.error(f"Error unloading model {self.model_id}: {e}")

  def predict(self, board_state: List[int], current_turn: int,
      turn_number: int) -> List[float]:
    """
        추론 수행

        Args:
            board_state: 게임 보드 상태
            current_turn: 현재 턴
            turn_number: 턴 번호

        Returns:
            List[float]: 확률 배열
        """
    input_tensor = self.preprocess_input(board_state, current_turn, turn_number)
    with torch.no_grad():
      output = self.model(input_tensor)
    return self.postprocess_output(output)

  def preprocess_input(self, board_state: List[int], current_turn: int,
      turn_number: int) -> torch.Tensor:
    """
        입력 전처리
        Args:
            board_state: 게임 보드 상태
            current_turn: 현재 턴
            turn_number: 턴 번호
        Returns:
            torch.Tensor: 전처리된 입력 텐서
        """
    # [current_turn, board_state...] 순서로 입력 텐서 생성
    x = [current_turn] + list(board_state)
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0)  # 배치 차원 추가
    return x.to(self.device)

  def postprocess_output(self, output: torch.Tensor) -> List[float]:
    """
        출력 후처리
        Args:
            output: 모델 출력 텐서
        Returns:
            List[float]: 후처리된 확률 배열
        """
    # JIT 모델이 반환하는 확률을 그대로 반환
    return output.squeeze(0).cpu().tolist()


class AIManager:
  """AI 매니저 클래스"""

  def __init__(self):
    """AI 매니저 초기화"""
    self.models: Dict[str, AIModel] = {}
    self.model_paths: Dict[str, str] = {}
    self.max_loaded_models = Config.MAX_LOADED_MODELS
    self.max_saved_models = Config.MAX_SAVED_MODELS
    self.models_dir = Path("models")
    self.logger = logging.getLogger(__name__)

    # 모델 디렉토리 생성
    self.models_dir.mkdir(exist_ok=True)

    # 메트릭용 추론 시간 리스트 (최근 100개 유지)
    self.inference_times: List[float] = []

  def load_model(self, request: ModelLoadRequest) -> ModelLoadResponse:
    """
        모델 로드 요청 처리

        Args:
            request: 모델 로드 요청

        Returns:
            ModelLoadResponse: 로드 응답
        """
    model_id = request.model_id
    model_url = request.model_url
    game_type = request.game_type
    try:
      if model_id in self.models:
        return ModelLoadResponse(
            request_id=request.request_id,
            timestamp=request.timestamp,
            game_id=request.game_id,
            model_id=model_id,
            success=True,
            message="Model already loaded.",
            ai_server_id=Config.SERVER_ID
        )
      model_path = self.download_model(model_url, model_id)
      ai_model = AIModel(model_id, model_path, game_type)
      if not ai_model.load():
        return ModelLoadResponse(
            request_id=request.request_id,
            timestamp=request.timestamp,
            game_id=request.game_id,
            model_id=model_id,
            success=False,
            message="Model load failed.",
            ai_server_id=Config.SERVER_ID
        )
      self.models[model_id] = ai_model
      self.model_paths[model_id] = model_path
      # LRU 캐시 관리
      if len(self.models) > self.max_loaded_models:
        self.cleanup_old_models()
      return ModelLoadResponse(
          request_id=request.request_id,
          timestamp=request.timestamp,
          game_id=request.game_id,
          model_id=model_id,
          success=True,
          message="Model loaded successfully.",
          ai_server_id=Config.SERVER_ID
      )
    except Exception as e:
      self.logger.error(f"Error loading model {model_id}: {e}")
      return ModelLoadResponse(
          request_id=request.request_id,
          timestamp=request.timestamp,
          game_id=request.game_id,
          model_id=model_id,
          success=False,
          message=str(e),
          ai_server_id=Config.SERVER_ID
      )

  def unload_model(self, model_id: str) -> bool:
    """
        모델 언로드

        Args:
            model_id: 모델 ID

        Returns:
            bool: 언로드 성공 여부
        """
    if model_id in self.models:
      try:
        self.models[model_id].unload()
        del self.models[model_id]
        del self.model_paths[model_id]
        return True
      except Exception as e:
        self.logger.error(f"Error unloading model {model_id}: {e}")
        return False
    return False

  def get_model(self, model_id: str) -> Optional[AIModel]:
    """
        모델 가져오기

        Args:
            model_id: 모델 ID

        Returns:
            Optional[AIModel]: 모델 객체 (없으면 None)
        """
    return self.models.get(model_id, None)

  def download_model(self, model_url: str, model_id: str) -> str:
    """
        모델 파일 다운로드

        Args:
            model_url: 모델 URL
            model_id: 모델 ID

        Returns:
            str: 다운로드된 파일 경로
        """
    parsed = urlparse(model_url)
    ext = os.path.splitext(parsed.path)[-1]
    local_path = str(self.models_dir / f"{model_id}{ext}")
    if os.path.exists(local_path):
      return local_path
    r = requests.get(model_url, stream=True)
    if r.status_code == 200:
      with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
          f.write(chunk)
      return local_path
    else:
      raise RuntimeError(f"Failed to download model from {model_url}")

  def inference(self, request: InferenceRequest) -> InferenceResponse:
    """AI 추론 수행"""
    start_time = time.time()

    try:
      self.logger.info(
        f"추론 시작: model_id={request.model_id}, game_type={request.game_type}")
      self.logger.info(
        f"보드 상태: {request.board_state}, 길이={len(request.board_state)}")

      # 모델 존재 확인
      if request.model_id not in self.models:
        error_msg = f"모델을 찾을 수 없음: {request.model_id}"
        self.logger.error(error_msg)
        response_time_ms = int((time.time() - start_time) * 1000)
        return self._create_inference_response(request, [], False, error_msg,
                                               response_time_ms)

      ai_model = self.models[request.model_id]  # AIModel 객체
      self.logger.info(f"모델 발견: {request.model_id}, 타입: {type(ai_model)}")

      # ✅ AIModel의 predict 메서드 사용 (올바른 방식)
      result = ai_model.predict(
          board_state=request.board_state,
          current_turn=request.current_turn,
          turn_number=request.turn_number
      )
      self.logger.info(f"추론 성공: 결과 길이={len(result)}")

      # 추론 시간 기록 (메트릭용)
      elapsed = time.time() - start_time
      self.inference_times.append(elapsed)
      if len(self.inference_times) > 100:  # 최근 100개만 유지
        self.inference_times.pop(0)

      response_time_ms = int((time.time() - start_time) * 1000)
      return self._create_inference_response(request, result, True, "",
                                             response_time_ms)

    except Exception as e:
      error_msg = f"추론 중 오류 발생: {str(e)}"
      self.logger.error(error_msg, exc_info=True)
      response_time_ms = int((time.time() - start_time) * 1000)
      return self._create_inference_response(request, [], False, error_msg,
                                             response_time_ms)

  def _create_inference_response(self, request: InferenceRequest,
      probabilities: List[float], success: bool, message: str = "",
      response_time_ms: int = 0) -> InferenceResponse:
    """
        추론 응답 생성

        Args:
            request: 원본 요청
            probabilities: 확률 리스트
            success: 성공 여부
            message: 메시지 (기본값: "")
            response_time_ms: 응답 시간 (ms 단위, 기본값: 0)

        Returns:
            InferenceResponse: 생성된 응답 객체
        """
    return InferenceResponse(
        request_id=request.request_id,
        timestamp=request.timestamp,
        game_id=request.game_id,
        model_id=request.model_id,
        probabilities=probabilities,
        success=success,
        message=message,
        response_time_ms=response_time_ms
    )

  def cleanup_old_models(self):
    """오래된 모델 정리"""
    # LRU 캐시 정책에 따라 오래된 모델 제거
    # 디스크에서도 파일 삭제
    if len(self.models) <= self.max_loaded_models:
      return
    # 단순히 가장 먼저 추가된 모델 제거 (OrderedDict 사용 권장, 여기선 dict pop)
    old_model_id = next(iter(self.models))
    self.unload_model(old_model_id)
    model_path = self.model_paths.get(old_model_id)
    if model_path and os.path.exists(model_path):
      os.remove(model_path)
    self.logger.info(f"Cleaned up old model: {old_model_id}")

  def get_model_info(self) -> Dict[str, Any]:
    """
        로드된 모델 정보 조회

        Returns:
            Dict[str, Any]: 모델 정보 딕셔너리
        """
    # 현재 로드된 모델 목록
    # 메모리 사용량
    # GPU 사용량
    pass

  def get_metrics(self) -> Dict[str, Any]:
    """
        메트릭 반환 (Redis 메트릭 시스템용)

        Returns:
            Dict[str, Any]: 평균 추론 시간 및 전체 추론 횟수
        """
    avg = sum(self.inference_times) / len(
      self.inference_times) if self.inference_times else 0
    return {
      "avg_inference_time": avg,
      "total_inferences": len(self.inference_times)
    }
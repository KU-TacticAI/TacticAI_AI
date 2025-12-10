from fastapi import FastAPI, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Gauge, Histogram, generate_latest, \
  REGISTRY, CONTENT_TYPE_LATEST
import psutil
import time
import os
import asyncio
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "your-admin-token")

# ============================================================
# Gateway API 메트릭
# ============================================================

# HTTP 요청 카운터
http_requests_total = Counter(
    'gateway_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# HTTP 응답 시간
http_request_duration_seconds = Histogram(
    'gateway_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# 활성 연결 수
active_connections = Gauge(
    'gateway_active_connections',
    'Number of active connections'
)

# ============================================================
# RabbitMQ 메트릭
# ============================================================

# 발행된 게임 요청 메시지
rabbitmq_game_requests_published = Counter(
    'gateway_rabbitmq_game_requests_published_total',
    'Total game request messages published to RabbitMQ',
    ['game_type', 'status']  # status: success, failed
)

# 게임 진행상황 메시지 소비
rabbitmq_game_progress_consumed = Counter(
    'gateway_rabbitmq_game_progress_consumed_total',
    'Total game progress messages consumed from RabbitMQ',
    ['game_id']
)

# RabbitMQ 연결 상태
rabbitmq_connection_status = Gauge(
    'gateway_rabbitmq_connection_status',
    'RabbitMQ connection status (1=connected, 0=disconnected)'
)

# RabbitMQ 발행 실패
rabbitmq_publish_failures = Counter(
    'gateway_rabbitmq_publish_failures_total',
    'Total RabbitMQ publish failures',
    ['reason']
)

# 메시지 발행 시간
rabbitmq_publish_duration_seconds = Histogram(
    'gateway_rabbitmq_publish_duration_seconds',
    'Time to publish message to RabbitMQ',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# ============================================================
# 게임 세션 메트릭
# ============================================================

# 활성 게임 세션
active_game_sessions = Gauge(
    'gateway_active_game_sessions',
    'Number of active game sessions',
    ['game_type']
)

# 생성된 게임 세션 (누적)
game_sessions_created = Counter(
    'gateway_game_sessions_created_total',
    'Total game sessions created',
    ['game_type']
)

# 완료된 게임 세션 (누적)
game_sessions_completed = Counter(
    'gateway_game_sessions_completed_total',
    'Total game sessions completed',
    ['game_type', 'result']  # result: finished, error, timeout
)

# 게임 진행상황 큐 생성
game_progress_queues_created = Counter(
    'gateway_game_progress_queues_created_total',
    'Total game progress queues created',
    ['status']  # status: success, failed
)

# 게임 진행상황 조회 횟수
game_progress_queries = Counter(
    'gateway_game_progress_queries_total',
    'Total game progress queries',
    ['game_id', 'player_id']
)

# ============================================================
# 시스템 메트릭
# ============================================================

# CPU 사용률
cpu_usage_percent = Gauge(
    'gateway_cpu_usage_percent',
    'CPU usage percentage'
)

# 메모리 사용량
memory_usage_bytes = Gauge(
    'gateway_memory_usage_bytes',
    'Memory usage in bytes'
)

# 메모리 사용률
memory_usage_percent = Gauge(
    'gateway_memory_usage_percent',
    'Memory usage percentage'
)

# 프로세스 스레드 수
process_threads = Gauge(
    'gateway_process_threads',
    'Number of threads'
)

# 업타임
uptime_seconds = Gauge(
    'gateway_uptime_seconds',
    'Uptime in seconds'
)


# ============================================================
# 메트릭 업데이트 함수들
# ============================================================

def record_http_request(method: str, endpoint: str, status: int,
    duration: float):
  """HTTP 요청 기록"""
  http_requests_total.labels(method=method, endpoint=endpoint,
                             status=status).inc()
  http_request_duration_seconds.labels(method=method,
                                       endpoint=endpoint).observe(duration)


def record_game_request_published(game_type: str, success: bool):
  """게임 요청 발행 기록"""
  status = "success" if success else "failed"
  rabbitmq_game_requests_published.labels(game_type=game_type,
                                          status=status).inc()


def record_game_progress_consumed(game_id: str, count: int = 1):
  """게임 진행상황 소비 기록"""
  rabbitmq_game_progress_consumed.labels(game_id=game_id).inc(count)


def update_rabbitmq_connection_status(connected: bool):
  """RabbitMQ 연결 상태 업데이트"""
  rabbitmq_connection_status.set(1 if connected else 0)


def record_rabbitmq_publish_failure(reason: str):
  """RabbitMQ 발행 실패 기록"""
  rabbitmq_publish_failures.labels(reason=reason).inc()


def record_rabbitmq_publish_duration(duration: float):
  """RabbitMQ 발행 시간 기록"""
  rabbitmq_publish_duration_seconds.observe(duration)


def update_active_game_sessions(game_type: str, count: int):
  """활성 게임 세션 수 업데이트"""
  active_game_sessions.labels(game_type=game_type).set(count)


def record_game_session_created(game_type: str):
  """게임 세션 생성 기록"""
  game_sessions_created.labels(game_type=game_type).inc()


def record_game_session_completed(game_type: str, result: str):
  """게임 세션 완료 기록"""
  game_sessions_completed.labels(game_type=game_type, result=result).inc()


def record_game_progress_queue_created(success: bool):
  """게임 진행상황 큐 생성 기록"""
  status = "success" if success else "failed"
  game_progress_queues_created.labels(status=status).inc()


def record_game_progress_query(game_id: str, player_id: str):
  """게임 진행상황 조회 기록"""
  game_progress_queries.labels(game_id=game_id, player_id=player_id).inc()


def increment_active_connections():
  """활성 연결 수 증가"""
  active_connections.inc()


def decrement_active_connections():
  """활성 연결 수 감소"""
  active_connections.dec()


# ============================================================
# 시스템 메트릭 업데이트
# ============================================================

_start_time = time.time()


def update_system_metrics():
  """시스템 메트릭 업데이트"""
  try:
    # CPU 사용률
    cpu_usage_percent.set(psutil.cpu_percent(interval=0.1))

    # 메모리
    memory = psutil.virtual_memory()
    memory_usage_bytes.set(memory.used)
    memory_usage_percent.set(memory.percent)

    # 프로세스 정보
    process = psutil.Process()
    process_threads.set(process.num_threads())

    # 업타임
    uptime_seconds.set(time.time() - _start_time)
  except Exception as e:
    logger.error(f"Failed to update system metrics: {e}")


# ============================================================
# 미들웨어: HTTP 요청 자동 추적
# ============================================================

async def metrics_middleware(request, call_next):
  """모든 HTTP 요청에 대해 자동으로 메트릭 기록"""
  start_time = time.time()

  # 연결 수 증가
  increment_active_connections()

  # 최근 요청 추적
  await track_request()

  try:
    response = await call_next(request)
    duration = time.time() - start_time

    # 메트릭 기록
    record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )

    return response

  except Exception as e:
    duration = time.time() - start_time
    record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status=500,
        duration=duration
    )
    raise

  finally:
    # 연결 수 감소
    decrement_active_connections()


# ============================================================
# 백그라운드 태스크: 주기적 업데이트
# ============================================================

async def update_metrics_periodically():
  """주기적으로 메트릭 업데이트"""
  while True:
    try:
      update_system_metrics()
      await asyncio.sleep(10)  # 10초마다 업데이트
    except asyncio.CancelledError:
      break
    except Exception as e:
      logger.error(f"Error in update_metrics_periodically: {e}")
      await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
  """앱 시작/종료 시 실행"""
  # 시작 시
  task = asyncio.create_task(update_metrics_periodically())
  yield
  # 종료 시
  task.cancel()


# ============================================================
# Prometheus 메트릭 엔드포인트
# ============================================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
  """토큰 인증"""
  if credentials.credentials != ADMIN_TOKEN:
    from fastapi import HTTPException, status
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token"
    )


def setup_metrics_endpoint(app: FastAPI):
  """FastAPI 앱에 메트릭 엔드포인트 추가"""

  @app.get("/metrics")
  async def metrics(token: str = Depends(verify_token)):
    """Prometheus 메트릭 엔드포인트"""
    # 최신 시스템 메트릭 업데이트
    update_system_metrics()

    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

  @app.get("/metrics/health")
  async def metrics_health():
    """메트릭 시스템 헬스 체크 (인증 불필요)"""
    return {
      "status": "healthy",
      "metrics_enabled": True,
      "timestamp": time.time()
    }


# ============================================================
# RabbitMQ 클라이언트 래퍼 (메트릭 추가)
# ============================================================

class MetricsRabbitMQWrapper:
  """RabbitMQ 클라이언트에 메트릭 추적을 추가하는 래퍼"""

  def __init__(self, client):
    self.client = client

  def publish_game_request(self, request):
    """게임 요청 발행 (메트릭 추가)"""
    start_time = time.time()
    try:
      result = self.client.publish_game_request(request)
      duration = time.time() - start_time

      # 메트릭 기록
      record_rabbitmq_publish_duration(duration)
      record_game_request_published(request.game_type, result)
      record_game_session_created(request.game_type)

      # 연결 상태 업데이트
      update_rabbitmq_connection_status(self.client.is_connected())

      return result
    except Exception as e:
      duration = time.time() - start_time
      record_rabbitmq_publish_duration(duration)
      record_game_request_published(request.game_type, False)
      record_rabbitmq_publish_failure(str(type(e).__name__))
      update_rabbitmq_connection_status(False)
      raise

  def create_game_progress_queue(self, game_id: str, player_ids: list):
    """게임 진행상황 큐 생성 (메트릭 추가)"""
    try:
      result = self.client.create_game_progress_queue(game_id, player_ids)
      record_game_progress_queue_created(result)
      return result
    except Exception as e:
      record_game_progress_queue_created(False)
      raise

  def get_game_progress_messages(self, game_id: str, player_id: str,
      limit: int = 10):
    """게임 진행상황 메시지 조회 (메트릭 추가)"""
    messages = self.client.get_game_progress_messages(game_id, player_id, limit)

    # 메트릭 기록
    record_game_progress_query(game_id, player_id)
    if messages:
      record_game_progress_consumed(game_id, len(messages))

    return messages

  def __getattr__(self, name):
    """나머지 메서드는 원본 클라이언트로 전달"""
    return getattr(self.client, name)


def wrap_rabbitmq_client_with_metrics(client):
  """RabbitMQ 클라이언트를 메트릭 래퍼로 감싸기"""
  return MetricsRabbitMQWrapper(client)


# ============================================================
# 실시간 메트릭 조회 함수
# ============================================================

# 최근 10초 요청 수를 추적하기 위한 전역 변수
_recent_requests = []
_recent_requests_lock = asyncio.Lock()


def get_realtime_gateway_metrics():
  """
  게이트웨이 자체의 실시간 메트릭 반환

  Returns:
    dict: 실시간 사용자 수, 최근 10초 요청 수, CPU 사용률 등
  """
  try:
    # 시스템 메트릭 최신화
    update_system_metrics()

    # Prometheus 메트릭에서 값 읽기
    active_users = int(active_connections._value.get())
    cpu_percent = cpu_usage_percent._value.get()
    memory_percent = memory_usage_percent._value.get()

    # 최근 10초 요청 수 계산
    recent_requests = get_recent_requests_count()

    # RabbitMQ 연결 상태
    rabbitmq_connected = rabbitmq_connection_status._value.get() == 1.0

    # 업타임
    uptime = int(time.time() - _start_time)

    return {
      "timestamp": time.time(),
      "activeUsers": active_users,
      "recentRequests": recent_requests,
      "cpuUsagePercent": round(cpu_percent, 2),
      "memoryUsagePercent": round(memory_percent, 2),
      "rabbitmqConnected": rabbitmq_connected,
      "uptimeSeconds": uptime
    }
  except Exception as e:
    logger.error(f"Failed to get realtime metrics: {e}", exc_info=True)
    return {
      "timestamp": time.time(),
      "activeUsers": 0,
      "recentRequests": 0,
      "cpuUsagePercent": 0.0,
      "memoryUsagePercent": 0.0,
      "rabbitmqConnected": False,
      "uptimeSeconds": 0,
      "error": str(e)
    }


async def track_request():
  """최근 요청을 추적 (미들웨어에서 호출)"""
  async with _recent_requests_lock:
    current_time = time.time()
    _recent_requests.append(current_time)

    # 10초 이전 요청 제거
    cutoff_time = current_time - 10
    while _recent_requests and _recent_requests[0] < cutoff_time:
      _recent_requests.pop(0)


def get_recent_requests_count():
  """최근 10초간 요청 수 반환"""
  current_time = time.time()
  cutoff_time = current_time - 10
  return sum(1 for t in _recent_requests if t >= cutoff_time)
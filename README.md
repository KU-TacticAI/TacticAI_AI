# TacticAI - AI 대전 플랫폼을 위한 분산 게임 서버 시스템

대규모 사용자가 동시 접속하여 **체스, 오셀로, 틱택토, 오목** 등 턴제 보드게임에서 AI 간 대전을 수행하는 **고신뢰·고확장 분산 게임 서버 시스템**입니다.

## 🎯 주요 특징

- **마이크로서비스 아키텍처(MSA)**: Game Server, AI Server, MQ Gateway 분리
- **Kubernetes 기반 오케스트레이션**: 자동 확장(HPA), 자가복구, 롤링 배포
- **RabbitMQ 비동기 메시징**: 서비스 간 느슨한 결합, 동적 모델 라우팅
- **GPU 추론 최적화**: NVIDIA MPS로 다중 AI Server Pod 운영
- **실시간 모니터링**: Prometheus + Grafana 통합 대시보드

## 🏗️ 아키텍처

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│ MQ Gateway  │────▶│  RabbitMQ   │
│   (React)   │     │  (FastAPI)  │     │  (Broker)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐              ┌─────▼─────┐              ┌─────▼─────┐
              │   Game    │              │    AI     │              │   Model   │
              │  Server   │◀────────────▶│  Server   │◀────────────▶│  Storage  │
              └───────────┘              └───────────┘              └───────────┘
```

## 📁 프로젝트 구조

```
TacticAI_AI/
├── ai_server/                # AI 추론 서버
│   ├── main.py               # 진입점 (AIServer)
│   ├── ai_manager.py         # 모델 관리 (AIManager, AIModel)
│   ├── rabbitmq_client.py    # RabbitMQ 클라이언트 (pika)
│   └── Dockerfile
├── game_server/              # 게임 서버
│   ├── main.py               # 진입점
│   ├── game_manager.py       # 게임 흐름 제어 (GameManager)
│   ├── rabbitmq_client.py    # RabbitMQ 클라이언트 (aio-pika)
│   ├── db.py                 # MongoDB/MySQL 연동
│   └── Game/BoardGame/       # 게임 로직
│       ├── BoardGame.py      # 추상 클래스
│       ├── Chess.py          # 체스
│       ├── Othello.py        # 오델로
│       ├── Omok.py           # 오목
│       └── TicTacToe.py      # 틱택토
├── mq_api_gateway/           # API 게이트웨이
│   ├── main.py               # FastAPI 앱
│   └── rabbitmq_client.py    # RabbitMQ 클라이언트
├── demo/                     # 프론트엔드 (React + Vite)
│   └── src/
│       ├── App.tsx           # 메인 앱
│       └── game/             # 게임 컴포넌트
├── helm/                     # Kubernetes Helm 차트
└── test/                     # K6 부하 테스트 스크립트
```

## 🚀 실행 방법

### 🐳 Docker 이미지 빌드 및 사설 레지스트리 푸시

```bash
# 사설 레지스트리 주소 (예시)
REGISTRY=172.31.35.34:5000

# AI Server 이미지 빌드 및 푸시
docker build -t ${REGISTRY}/tacticai/ai-server:latest ./ai_server
docker push ${REGISTRY}/tacticai/ai-server:latest

# Game Server 이미지 빌드 및 푸시
docker build -t ${REGISTRY}/tacticai/game-server:latest ./game_server
docker push ${REGISTRY}/tacticai/game-server:latest

# MQ API Gateway 이미지 빌드 및 푸시
docker build -t ${REGISTRY}/tacticai/mq-api-gateway:latest ./mq_api_gateway
docker push ${REGISTRY}/tacticai/mq-api-gateway:latest

# 프론트엔드 이미지 빌드 및 푸시
docker build -t ${REGISTRY}/tacticai/demo:latest ./demo
docker push ${REGISTRY}/tacticai/demo:latest
```

> **참고**: 사설 레지스트리가 HTTP인 경우 Docker에 insecure-registry 설정 필요:
> ```json
> // /etc/docker/daemon.json
> { "insecure-registries": ["172.31.35.34:5000"] }
> ```

### 🎮 NVIDIA GPU Operator 설치 및 MPS 설정 (Linux)

GPU 1개를 여러 AI Server Pod에서 공유하기 위해 NVIDIA GPU Operator와 MPS 설정이 필요합니다.

**1. 사전 요구사항 확인 (GPU 노드에서 실행):**

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# containerd 설정 (nvidia-container-toolkit 필요)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

**2. Helm을 통한 GPU Operator 설치:**

```bash
# Helm repo 추가
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# GPU Operator 네임스페이스 생성 및 설치
kubectl create namespace gpu-operator

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set mig.strategy=single
```

> **참고**: 호스트에 이미 NVIDIA 드라이버가 설치된 경우 `driver.enabled=false` 설정

**3. MPS 설정을 위한 ConfigMap 적용:**

```bash
kubectl apply -f helm/mps-config.yaml
```

**4. ClusterPolicy에 MPS 설정 적용:**

```bash
kubectl patch clusterpolicy/cluster-policy \
  -n gpu-operator \
  --type merge \
  -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'
```

**5. 설정 적용 확인:**

```bash
# GPU Operator Pod 상태 확인
kubectl get pods -n gpu-operator

# GPU 노드 리소스 확인 (nvidia.com/gpu: 4 표시)
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu
```

### ☸️ Helm 차트로 배포

```bash
# Helm 차트로 배포
helm install tacticai ./helm -n tacticai --create-namespace

# Pod 상태 확인
kubectl get pods -n tacticai
```

### 프론트엔드 개발 서버

**1. MQ Gateway 프록시 설정 (`demo/vite.config.ts`):**

Kubernetes에 배포된 MQ Gateway 서버 주소를 설정합니다:

```typescript
proxy: {
  '/ai': {
    target: 'http://<MQ_GATEWAY_NODE_IP>:<NODE_PORT>',  // 예: http://3.35.xxx.xxx:30080
    changeOrigin: true,
    secure: false,
  }
}
```

> **MQ Gateway NodePort 확인:**
> ```bash
> kubectl get svc -n tacticai | grep mq-api-gateway
> ```

**2. 개발 서버 실행:**

```bash
cd demo
npm install
npm run dev
```

## 🔧 환경 변수

`.env` 파일 생성:

```env
# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Database
MONGODB_URI=mongodb://localhost:27017
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=password

# AI Server
MODEL_STORAGE_PATH=./models
DEVICE=cuda  # or cpu
```

## 📊 모니터링

각 서비스는 Prometheus 형식의 메트릭을 노출합니다:

| 서비스 | 엔드포인트 | 주요 메트릭 |
|--------|-----------|------------|
| **Gateway** | `/metric/realtime` | Active Users, Traffic Load |
| **AI Server** | `/metric/realtime/ai-performance` | Inference Latency, Pod Status |
| **Game Server** | `/metric/realtime/game-stats` | Active Matches, Waiting Users |

## 🧪 테스트

### K6 부하 테스트

```bash
# K6 설치 후 실행
k6 run test/full-game-scenario.js
```

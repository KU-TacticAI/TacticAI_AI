#!/bin/bash
set -e

echo "=== RabbitMQ Setup Script ==="

# RabbitMQ 컨테이너 이름
RABBITMQ_CONTAINER="tacticai-rabbitmq"
RABBITMQ_IMAGE="rabbitmq:3.13.7-management"

# 환경 변수에서 읽기 (기본값 설정)
RABBITMQ_USER="${RABBITMQ_USER:-admin}"
RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-admin}"
RABBITMQ_PORT="${RABBITMQ_PORT:-5672}"
RABBITMQ_MANAGEMENT_PORT="${RABBITMQ_MANAGEMENT_PORT:-15672}"

echo "1. Pulling RabbitMQ image (version 3.13.7)..."
docker pull $RABBITMQ_IMAGE

echo "2. Stopping existing RabbitMQ container (if any)..."
docker stop $RABBITMQ_CONTAINER 2>/dev/null || true
docker rm $RABBITMQ_CONTAINER 2>/dev/null || true

echo "3. Starting RabbitMQ container..."
docker run -d \
  --name $RABBITMQ_CONTAINER \
  -p $RABBITMQ_PORT:5672 \
  -p $RABBITMQ_MANAGEMENT_PORT:15672 \
  -e RABBITMQ_DEFAULT_USER=$RABBITMQ_USER \
  -e RABBITMQ_DEFAULT_PASS=$RABBITMQ_PASSWORD \
  $RABBITMQ_IMAGE

echo "4. Waiting for RabbitMQ to be ready..."
sleep 15

# RabbitMQ가 준비될 때까지 대기
for i in {1..30}; do
  if docker exec $RABBITMQ_CONTAINER rabbitmqctl status >/dev/null 2>&1; then
    echo "RabbitMQ is ready!"
    break
  fi
  echo "Waiting for RabbitMQ... ($i/30)"
  sleep 2
done

echo "5. Setting up RabbitMQ users and permissions..."

# testuser 생성 (monitoring 태그)
docker exec $RABBITMQ_CONTAINER rabbitmqctl add_user testuser testuser || true
docker exec $RABBITMQ_CONTAINER rabbitmqctl set_user_tags testuser monitoring
docker exec $RABBITMQ_CONTAINER rabbitmqctl set_permissions -p / testuser ".*" ".*" ".*"

# topic permissions for testuser
docker exec $RABBITMQ_CONTAINER rabbitmqctl set_topic_permissions -p / testuser "" ".*" ".*"

echo "6. Setting up RabbitMQ exchanges..."

# Exchange 생성 (순서: game_request, model_load, game_progress, response, ai_inference)
docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare exchange \
  name=game_request_exchange type=direct durable=true

docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare exchange \
  name=model_load_exchange type=direct durable=true

docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare exchange \
  name=game_progress_exchange type=topic durable=true

docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare exchange \
  name=response_exchange type=topic durable=true

docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare exchange \
  name=ai_inference_exchange type=topic durable=true

echo "7. Setting up RabbitMQ queues..."

# Queue 생성
docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare queue \
  name=game_request_queue durable=true

docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare queue \
  name=model_load_queue durable=true

echo "8. Setting up RabbitMQ bindings..."

# Binding 생성
# game_request_exchange -> game_request_queue (routing_key: game_request_queue)
docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare binding \
  source=game_request_exchange destination=game_request_queue routing_key=game_request_queue

# model_load_exchange -> model_load_queue (routing_key: model_load)
docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare binding \
  source=model_load_exchange destination=model_load_queue routing_key=model_load

# model_load_exchange -> model_load_queue (routing_key: model_load_queue)
docker exec $RABBITMQ_CONTAINER rabbitmqadmin declare binding \
  source=model_load_exchange destination=model_load_queue routing_key=model_load_queue

echo "9. Setting up RabbitMQ policies..."

# game_progress 큐에 대한 만료 정책 (60초)
docker exec $RABBITMQ_CONTAINER rabbitmqctl set_policy \
  q-exp "^game_progress.*" '{"expires":60000}' --apply-to queues --priority 0

echo "10. RabbitMQ setup completed successfully!"
echo ""
echo "=== RabbitMQ Configuration Summary ==="
echo "Version: 3.13.7"
echo "Management UI: http://localhost:$RABBITMQ_MANAGEMENT_PORT"
echo ""
echo "Users:"
echo "  - admin (administrator): $RABBITMQ_USER"
echo "  - guest (administrator): guest"
echo "  - testuser (monitoring): testuser"
echo ""
echo "Exchanges:"
echo "  - game_request_exchange (direct)"
echo "  - model_load_exchange (direct)"
echo "  - game_progress_exchange (topic)"
echo "  - response_exchange (topic)"
echo "  - ai_inference_exchange (topic)"
echo ""
echo "Queues:"
echo "  - game_request_queue (durable)"
echo "  - model_load_queue (durable)"
echo ""
echo "Policies:"
echo "  - q-exp: game_progress queues expire after 60 seconds"
echo ""
echo "AMQP Connection: amqp://$RABBITMQ_USER:$RABBITMQ_PASSWORD@localhost:$RABBITMQ_PORT/"

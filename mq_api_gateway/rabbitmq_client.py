#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync-compatible RabbitMQ client for MQ API Gateway.

This file exposes the same synchronous function names and return types as
the original `pika`-based implementation, but internally uses `aio-pika`.
It runs an asyncio event loop in a background thread and dispatches
coroutines with `asyncio.run_coroutine_threadsafe`, so callers don't need
to be changed immediately.
"""

import asyncio
import threading
import json
import logging
from typing import Optional, List

from aio_pika import connect_robust, Message, DeliveryMode, ExchangeType
from aio_pika.abc import AbstractRobustConnection, AbstractRobustChannel

from config import Config
from message_schemas import GameRequest

logger = logging.getLogger(__name__)


class _AsyncRabbit:
    """Internal aio-pika async client."""

    def __init__(self, config: Config):
        self.config = config
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel: Optional[AbstractRobustChannel] = None
        # consumers: queue_name -> consumer_tag
        self._consumers = {}
        # in-memory message buffers: queue_name -> list of messages
        self._buffers = {}
        # lock for buffer access
        self._buffer_locks = {}
        # last used timestamp for consumer idle management
        self._last_used = {}
        # background task to stop idle consumers
        self._idle_task = None
        # idle timeout in seconds (configurable via Config, fallback to 30s)
        self._idle_timeout = getattr(self.config, 'GAME_PROGRESS_CONSUMER_IDLE_TIMEOUT', 30)

    async def connect(self):
        if self.connection and not self.connection.is_closed:
            return True

        params = {}
        if getattr(self.config, 'RABBITMQ_HOST', None):
            params['host'] = self.config.RABBITMQ_HOST
        if getattr(self.config, 'RABBITMQ_PORT', None):
            try:
                params['port'] = int(self.config.RABBITMQ_PORT)
            except Exception:
                params['port'] = self.config.RABBITMQ_PORT
        if getattr(self.config, 'RABBITMQ_USER', None):
            params['login'] = self.config.RABBITMQ_USER
        if getattr(self.config, 'RABBITMQ_PASSWORD', None):
            params['password'] = self.config.RABBITMQ_PASSWORD

        self.connection = await connect_robust(**params)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)

        # ensure exchanges exist
        await self.channel.declare_exchange(self.config.GAME_REQUEST_EXCHANGE, ExchangeType.DIRECT, durable=True)
        await self.channel.declare_exchange(self.config.GAME_PROGRESS_EXCHANGE, ExchangeType.TOPIC, durable=True)
        return True

    async def close(self):
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
        except Exception:
            pass
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
        except Exception:
            pass

    async def publish_game_request(self, request: GameRequest) -> bool:
        try:
            await self.connect()
            exchange = await self.channel.declare_exchange(self.config.GAME_REQUEST_EXCHANGE, ExchangeType.DIRECT, durable=True)
            body = request.model_dump_json().encode()
            message = Message(body, delivery_mode=DeliveryMode.PERSISTENT, content_type='application/json')
            await exchange.publish(message, routing_key=self.config.GAME_REQUEST_QUEUE)
            logger.info(f"게임 요청 발행 성공: {request.game_id}")
            return True
        except Exception:
            logger.exception("Async publish_game_request failed")
            return False

    async def create_game_progress_queue(self, game_id: str, player_ids: List[str]) -> bool:
        try:
            await self.connect()
            routing_key = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}.{game_id}"
            exchange = await self.channel.declare_exchange(self.config.GAME_PROGRESS_EXCHANGE, ExchangeType.TOPIC, durable=True)
            for player_id in player_ids:
                queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}_{player_id}"
                queue = await self.channel.declare_queue(queue_name, durable=True)
                await queue.bind(exchange, routing_key=routing_key)
                logger.info(f"게임 진행상황 큐 생성 완료: {queue_name}")
            return True
        except Exception:
            logger.exception("Async create_game_progress_queue failed")
            return False

    async def get_game_progress_messages(self, game_id: str, player_id: str, limit: int = 10) -> list:
        if not game_id or not player_id:
            return []
        try:
            # Consumer-based reading handled by background consumers; sync shim will read from buffers.
            # This async path is not used in the shim. Return empty list if called directly.
            return []
        except Exception:
            logger.exception("Async get_game_progress_messages failed")
            return []

    async def delete_game_progress_queue(self, game_id: str, player_id: str) -> bool:
        try:
            await self.connect()
            queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}_{player_id}"
            try:
                queue = await self.channel.declare_queue(queue_name, passive=True)
            except Exception:
                logger.info(f"큐가 존재하지 않아 삭제할 필요 없음: {queue_name}")
                return True
            await queue.delete()
            logger.info(f"게임 진행상황 큐 삭제 완료: {queue_name}")
            return True
        except Exception:
            logger.exception("Async delete_game_progress_queue failed")
            return False

    async def _consumer_callback(self, queue_name: str, incoming):
        try:
            data = json.loads(incoming.body.decode())
        except Exception:
            logger.exception("consumer: JSON parse failed")
            try:
                await incoming.nack(requeue=False)
            except Exception:
                pass
            return

        # append to buffer
        buf = self._buffers.setdefault(queue_name, [])
        lock = self._buffer_locks.setdefault(queue_name, asyncio.Lock())
        async with lock:
            buf.append({
                'index': len(buf) + 1,
                'queue': queue_name,
                'data': data,
                'delivery_tag': None,
            })
        try:
            await incoming.ack()
        except Exception:
            pass

    async def start_consumer_for_queue(self, queue_name: str):
        await self.connect()
        if queue_name in self._consumers:
            return True
        try:
            queue = await self.channel.declare_queue(queue_name, durable=True)
            # create callback wrapper
            async def _cb(incoming):
                await self._consumer_callback(queue_name, incoming)

            consumer_tag = await queue.consume(_cb)
            self._consumers[queue_name] = consumer_tag
            # mark last used
            self._last_used[queue_name] = asyncio.get_event_loop().time()
            # ensure idle stopper running
            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._idle_consumer_worker())
            # initialize buffer/lock
            self._buffers.setdefault(queue_name, [])
            self._buffer_locks.setdefault(queue_name, asyncio.Lock())
            return True
        except Exception:
            logger.exception(f"start_consumer_for_queue failed: {queue_name}")
            return False

    async def stop_consumer_for_queue(self, queue_name: str):
        try:
            if queue_name not in self._consumers:
                return True
            consumer_tag = self._consumers.pop(queue_name)
            queue = await self.channel.declare_queue(queue_name, passive=True)
            await queue.cancel(consumer_tag)
            # clear buffer
            self._buffers.pop(queue_name, None)
            self._buffer_locks.pop(queue_name, None)
            self._last_used.pop(queue_name, None)
            return True
        except Exception:
            logger.exception(f"stop_consumer_for_queue failed: {queue_name}")
            return False

    async def _idle_consumer_worker(self):
        """Background task that stops consumers which have been idle for > idle_timeout."""
        try:
            while True:
                now = asyncio.get_event_loop().time()
                to_stop = []
                for q, last in list(self._last_used.items()):
                    # if buffer empty and idle time exceeded -> stop
                    buf = self._buffers.get(q, [])
                    if (now - last) > self._idle_timeout and len(buf) == 0:
                        to_stop.append(q)

                for q in to_stop:
                    try:
                        await self.stop_consumer_for_queue(q)
                        logger.info(f"idle consumer stopped: {q}")
                    except Exception:
                        logger.exception(f"failed stopping idle consumer: {q}")

                # sleep a short while
                await asyncio.sleep(min(5, max(1, int(self._idle_timeout / 4))))
                # exit early if no consumers remain
                if not self._consumers:
                    break
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("idle_consumer_worker crashed")


class RabbitMQClient:
    """Sync-compatible interface backed by aio-pika running in background loop."""

    def __init__(self, config: Config):
        self.config = config
        self._async_client = _AsyncRabbit(config)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro, timeout: float = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except Exception:
            logger.exception("_run_coro exception")
            return None

    def connect(self) -> bool:
        res = self._run_coro(self._async_client.connect())
        return bool(res)

    def setup_topology(self):
        # topology is prepared by async connect
        return

    def publish_game_request(self, request: GameRequest) -> bool:
        res = self._run_coro(self._async_client.publish_game_request(request))
        return bool(res)

    def create_game_progress_queue(self, game_id: str, player_ids: List[str]) -> bool:
        res = self._run_coro(self._async_client.create_game_progress_queue(game_id, player_ids))
        if not res:
            return False
        return True

    def get_game_progress_messages(self, game_id: str, player_id: str, limit: int = 10) -> list:
        queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}_{player_id}"
        # fetch from in-memory buffer
        # use asyncio.run_coroutine_threadsafe to acquire buffer lock safely
        try:
            async def _ensure_consumer_and_drain():
                # start consumer lazily if not already
                if queue_name not in self._async_client._consumers:
                    await self._async_client.start_consumer_for_queue(queue_name)

                # update last-used to keep consumer alive
                self._async_client._last_used[queue_name] = asyncio.get_event_loop().time()

                lock = self._async_client._buffer_locks.setdefault(queue_name, asyncio.Lock())
                async with lock:
                    buf = self._async_client._buffers.setdefault(queue_name, [])
                    out = buf[:limit]
                    # remove drained
                    del buf[:len(out)]
                    return out

            future = asyncio.run_coroutine_threadsafe(_ensure_consumer_and_drain(), self._loop)
            result = future.result(timeout=2)
            return result if isinstance(result, list) else []
        except Exception:
            logger.exception("get_game_progress_messages buffer drain failed")
            return []

    def delete_game_progress_queue(self, game_id: str, player_id: str) -> bool:
        res = self._run_coro(self._async_client.delete_game_progress_queue(game_id, player_id))
        # stop consumer as well
        queue_name = f"{self.config.GAME_PROGRESS_QUEUE_PREFIX}_{game_id}_{player_id}"
        try:
            self._run_coro(self._async_client.stop_consumer_for_queue(queue_name))
        except Exception:
            pass
        return bool(res)

    def is_connected(self) -> bool:
        conn = getattr(self._async_client, 'connection', None)
        return conn is not None and not getattr(conn, 'is_closed', False)

    def disconnect(self):
        try:
            self._run_coro(self._async_client.close(), timeout=5)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def __enter__(self):
        if not self.connect():
            raise RuntimeError("RabbitMQ 연결 실패")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# Singleton instance
_rabbitmq_client: Optional[RabbitMQClient] = None


def get_rabbitmq_client() -> RabbitMQClient:
    global _rabbitmq_client

    if _rabbitmq_client is None:
        config = Config()
        _rabbitmq_client = RabbitMQClient(config)

    return _rabbitmq_client


def close_rabbitmq_client():
    global _rabbitmq_client

    if _rabbitmq_client:
        _rabbitmq_client.disconnect()
        _rabbitmq_client = None
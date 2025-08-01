import asyncio
import threading

from collections.abc import Callable
from typing import Any

from memos.dependency import require_python_package
from memos.log import get_logger
from memos.mem_scheduler.modules.base import BaseSchedulerModule


logger = get_logger(__name__)


class RedisSchedulerModule(BaseSchedulerModule):
    @require_python_package(
        import_name="redis",
        install_command="pip install redis",
        install_link="https://redis.readthedocs.io/en/stable/",
    )
    def __init__(self):
        """
        intent_detector: Object used for intent recognition (such as the above IntentDetector)
        scheduler: The actual scheduling module/interface object
        trigger_intents: The types of intents that need to be triggered (list)
        """
        super().__init__()

        # settings for redis
        self.redis_host: str = None
        self.redis_port: int = None
        self.redis_db: int = None
        self._redis_conn = None
        self.query_list_capacity = 1000

        self._redis_listener_running = False
        self._redis_listener_thread: threading.Thread | None = None
        self._redis_listener_loop: asyncio.AbstractEventLoop | None = None

    @property
    def redis(self) -> Any:
        return self._redis_conn

    @redis.setter
    def redis(self, value: Any) -> None:
        self._redis_conn = value

    def initialize_redis(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
        import redis

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        try:
            logger.debug(f"Connecting to Redis at {redis_host}:{redis_port}/{redis_db}")
            self._redis_conn = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=self.redis_db, decode_responses=True
            )
            # test conn
            if not self._redis_conn.ping():
                logger.error("Redis connection failed")
        except redis.ConnectionError as e:
            self._redis_conn = None
            logger.error(f"Redis connection error: {e}")
        self._redis_conn.xtrim("user:queries:stream", self.query_list_capacity)
        return self._redis_conn

    async def redis_add_message_stream(self, message: dict):
        logger.debug(f"add_message_stream: {message}")
        return self._redis_conn.xadd("user:queries:stream", message)

    async def redis_consume_message_stream(self, message: dict):
        logger.debug(f"consume_message_stream: {message}")

    def _redis_run_listener_async(self, handler: Callable):
        """Run the async listener in a separate thread"""
        self._redis_listener_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._redis_listener_loop)

        async def listener_wrapper():
            try:
                await self.__redis_listen_query_stream(handler)
            except Exception as e:
                logger.error(f"Listener thread error: {e}")
            finally:
                self._redis_listener_running = False

        self._redis_listener_loop.run_until_complete(listener_wrapper())

    async def __redis_listen_query_stream(
        self, handler=None, last_id: str = "$", block_time: int = 2000
    ):
        """Internal async stream listener"""
        import redis

        self._redis_listener_running = True
        while self._redis_listener_running:
            try:
                # Blocking read for new messages
                messages = self.redis.xread(
                    {"user:queries:stream": last_id}, count=1, block=block_time
                )

                if messages:
                    for _, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            try:
                                print(f"deal with message_data {message_data}")
                                await handler(message_data)
                                last_id = message_id
                            except Exception as e:
                                logger.error(f"Error processing message {message_id}: {e}")

            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
                self._redis_conn = None  # Force reconnection
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(1)

    def redis_start_listening(self, handler: Callable | None = None):
        """Start the Redis stream listener in a background thread"""
        if self._redis_listener_thread and self._redis_listener_thread.is_alive():
            logger.warning("Listener is already running")
            return

        if handler is None:
            handler = self.redis_consume_message_stream

        self._redis_listener_thread = threading.Thread(
            target=self._redis_run_listener_async,
            args=(handler,),
            daemon=True,
            name="RedisListenerThread",
        )
        self._redis_listener_thread.start()
        logger.info("Started Redis stream listener thread")

    def redis_stop_listening(self):
        """Stop the listener thread gracefully"""
        self._redis_listener_running = False
        if self._redis_listener_thread and self._redis_listener_thread.is_alive():
            self._redis_listener_thread.join(timeout=5.0)
            if self._redis_listener_thread.is_alive():
                logger.warning("Listener thread did not stop gracefully")
        logger.info("Redis stream listener stopped")

    def redis_close(self):
        """Close Redis connection"""
        if self._redis_conn is not None:
            self._redis_conn.close()
            self._redis_conn = None

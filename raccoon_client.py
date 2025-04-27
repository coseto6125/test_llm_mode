from collections import deque
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from time import perf_counter
from typing import Any, AsyncGenerator, Self

import aiohttp
import orjson
from loguru import logger

from config import RACCOONAI_API_BASE_URL, RACCOONAI_API_STAGING_API_KEY


class RaccoonAIClient:
    __slots__ = ("base_url", "api_path", "headers", "session", "own_session", "chat_messages", "_api_url")

    def __init__(self, base_url: str | None = None, api_path: str | None = None) -> None:
        self.base_url = base_url or RACCOONAI_API_BASE_URL
        self.api_path = api_path or "/api/v2/raccoon_ai_core/chat"
        self.api_path = f"/{self.api_path.lstrip('/')}" if self.api_path else self.api_path
        self._api_url = f"{self.base_url}{self.api_path}"

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {RACCOONAI_API_STAGING_API_KEY}",
        }
        self.session = None
        self.own_session = False
        self.chat_messages: deque[dict[str, Any]] = deque(maxlen=100)

        logger.debug(f"初始化 RaccoonAIClient, API 端點: {self._api_url}")

    async def __aenter__(self) -> Self:
        if self.session is None:
            self.session = self._create_session()
            self.own_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._close_session_if_owned()

    async def _close_session_if_owned(self) -> None:
        if self.own_session and self.session is not None:
            await self.session.close()
            self.session = None
            self.own_session = False

    def reset_chat_history(self) -> None:
        self.chat_messages.clear()

    @staticmethod
    def _create_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            json_serialize=lambda obj: orjson.dumps(obj).decode("utf-8"),
            connector=aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                ssl=False,
            ),
            timeout=aiohttp.ClientTimeout(total=30),
        )

    def _get_or_create_session(self) -> tuple[aiohttp.ClientSession, bool]:
        if self.session is not None:
            return self.session, False

        return self._create_session(), True

    @staticmethod
    def _create_user_message(message: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"text": message, "type": "text"}]}

    async def chat(
        self,
        message: str,
        brand_id: str = "66",
        reset_history: bool = False,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        發送異步請求到 Raccoon AI 聊天 API

        Args:
            message: 用戶訊息
            brand_id: 品牌 ID
            reset_history: 是否重置對話歷史
            history: 提供自訂對話歷史

        Returns:
            API 響應
        """
        if reset_history:
            self.reset_chat_history()
        if history:
            self.chat_messages = deque(history, maxlen=100)

        user_message = self._create_user_message(message)
        self.chat_messages.append(user_message)

        # 構建正確的 payload 格式（字典而非字串）
        payload = {
            "chat_history": list(self.chat_messages), 
            "brand_id": brand_id
        }
        
        logger.debug(f"發送請求到 API: {self._api_url}, 品牌ID: {brand_id}")

        session_to_use, need_close = self._get_or_create_session()

        try:
            start_time = perf_counter()
            start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            async with session_to_use.post(
                self._api_url,
                headers=self.headers,
                json=payload,  # 使用正確格式的 payload
                raise_for_status=False,
            ) as response:
                end_time = perf_counter()
                end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                date_header = response.headers.get("Date")
                dt = parsedate_to_datetime(date_header) + timedelta(hours=8) if date_header else datetime.now()
                logger.debug(f"Response received at: {dt}")

                if response.status != 200:
                    error_text = await response.text()
                    logger.exception(f"Raccoon AI API 錯誤: {response.status} - {error_text}")
                    raise Exception(
                        message=error_text,
                        status_code=response.status,
                        provider="Raccoon AI",
                    )

                result = await response.json(loads=orjson.loads)
                result["response_time"] = end_time - start_time
                result["request_data"] = {
                    "request_id": response.headers.get("x-amzn-RequestId"),
                    "request_start": start_date,
                    "request_end_local": end_date,
                    "request_end": str(dt),
                }
                return result
        finally:
            if need_close:
                await session_to_use.close()

    async def stream_chat(
        self,
        message: str,
        brand_id: str = "66",
        reset_history: bool = False,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        發送異步請求到 Raccoon AI 聊天 API 並處理串流響應

        Args:
            message: 用戶訊息
            brand_id: 品牌 ID
            reset_history: 是否重置對話歷史
            history: 提供自訂對話歷史

        Yields:
            從 API 串流接收到的文本塊
        """
        if reset_history:
            self.reset_chat_history()
        if history:
            self.chat_messages = deque(history, maxlen=100)

        payload = message

        payload |= {"stream": True}

        logger.debug(f"發送串流請求到 API: {self._api_url}, 品牌ID: {brand_id}")

        stream_headers = self.headers.copy()
        stream_headers["Accept"] = "text/event-stream"

        session_to_use, need_close = self._get_or_create_session()

        try:
            start_time = perf_counter()

            async with session_to_use.post(
                self._api_url, headers=stream_headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.exception(f"Raccoon AI API (串流) 錯誤: {response.status} - {error_text}")
                    raise Exception(
                        message=error_text,
                        status_code=response.status,
                        provider="Raccoon AI",
                    )

                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    line = memoryview(chunk).tobytes().decode("utf-8")
                    buffer += line

                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)

                        if not event_str.startswith("data:"):
                            if event_str.strip():
                                logger.debug(f"接收到非 data 的 SSE 事件: {event_str}")
                            continue

                        try:
                            data_json = event_str[5:].strip()  # 5 = len("data:")

                            if data_json == "[DONE]":
                                logger.debug("接收到串流結束標記 [DONE]")
                                return

                            data = orjson.loads(data_json)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk

                        except orjson.JSONDecodeError:
                            logger.warning(f"無法解析 SSE data: {data_json}")
                        except Exception as e:
                            logger.exception(f"處理 SSE 事件時發生錯誤: {e}")

                if buffer and buffer.strip().startswith("data:"):
                    data_json = buffer.strip()[5:].strip()
                    if data_json and data_json != "[DONE]":
                        try:
                            data = orjson.loads(data_json)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                        except (orjson.JSONDecodeError, Exception):
                            pass

                end_time = perf_counter()
                logger.debug(f"Raccoon AI API (串流) 請求完成，耗時: {end_time - start_time:.4f} 秒")

        except aiohttp.ClientError as e:
            logger.exception("連接 Raccoon AI API (串流) 時發生錯誤")
            raise Exception(message=f"Connection error: {e}", status_code=503, provider="Raccoon AI") from e
        except Exception as e:
            logger.exception("處理 Raccoon AI API (串流) 時發生未知錯誤")
            yield f"Error during streaming: {str(e)}"
        finally:
            if need_close:
                await session_to_use.close()

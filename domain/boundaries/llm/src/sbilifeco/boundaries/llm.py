from typing import Protocol, AsyncGenerator
from pydantic import BaseModel
from sbilifeco.models.base import Response


class ChatMessage(BaseModel):
    role: str
    content: str


class ILLM(Protocol):
    async def generate_reply(self, context: str) -> Response[str]:
        raise NotImplementedError()

    async def generate_streamed_reply(
        self, request_id: str, context: str
    ) -> Response[AsyncGenerator[str, None]]:
        raise NotImplementedError()

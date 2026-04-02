from typing import Protocol, AsyncGenerator
from pydantic import BaseModel
from sbilifeco.models.base import Response
from uuid import uuid4


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    """Represents a request to the LLM service."""

    request_id: str = str(uuid4())
    """Unique identifier for the request."""

    context: str = ""
    """Context for the LLM request."""

    randomness: float = 0.0
    """Randomness factor for the LLM response. Is a value between 0 and 1, where higher values result in more random responses."""


class ILLM(Protocol):
    async def generate_reply(self, context: str) -> Response[str]:
        raise NotImplementedError()

    async def generate_streamed_reply(
        self,
        request: LLMRequest,
    ) -> Response[AsyncGenerator[str, None]]:
        raise NotImplementedError()

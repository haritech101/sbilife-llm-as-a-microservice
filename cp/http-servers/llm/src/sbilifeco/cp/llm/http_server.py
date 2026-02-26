from __future__ import annotations
from typing import Annotated, AsyncGenerator
from traceback import format_exc
from sbilifeco.boundaries.llm import ILLM, ChatMessage
from sbilifeco.models.base import Response
from sbilifeco.cp.common.http.server import HttpServer
from sbilifeco.cp.llm.paths import Paths, LLMQuery
from sbilifeco.boundaries.llm import ChatMessage
from fastapi import Path, Body
from fastapi.responses import StreamingResponse, PlainTextResponse


class LLMHttpServer(HttpServer):
    def __init__(self):
        HttpServer.__init__(self)
        self.llm: ILLM
        self.streams: dict[str, AsyncGenerator[str, None]] = {}

    def set_llm(self, llm: ILLM) -> LLMHttpServer:
        self.llm = llm
        return self

    async def listen(self) -> None:
        await HttpServer.listen(self)

    async def stop(self) -> None:
        await HttpServer.stop(self)

    def build_routes(self) -> None:
        @self.post(Paths.QUERIES)
        async def generate_query(query: LLMQuery) -> Response[str]:
            try:
                return await self.llm.generate_reply(query.context)
            except Exception as e:
                return Response.error(e)

        @self.post(Paths.STREAMS)
        async def generate_stream(
            request_id: Annotated[str, Path()], query: Annotated[LLMQuery, Body()]
        ):
            try:
                response_with_stream = await self.llm.generate_streamed_reply(
                    request_id, query.context
                )
                if not response_with_stream.is_success:
                    return PlainTextResponse(
                        response_with_stream.message,
                        status_code=response_with_stream.code,
                    )
                elif response_with_stream.payload is None:
                    return PlainTextResponse(
                        "LLM response is inexplicably empty",
                        status_code=500,
                    )
                self.streams[request_id] = response_with_stream.payload

                async def stream_llm_reply(
                    request_id: str,
                ) -> AsyncGenerator[str, None]:
                    async for chunk in self.streams[request_id]:
                        yield chunk

                    await self.streams[request_id].aclose()
                    del self.streams[request_id]

                return StreamingResponse(
                    stream_llm_reply(request_id),
                    media_type="text/markdown",
                )
            except Exception as e:
                message = f"Error while generating stream out of LLM response: {e}"
                print(message)
                print(format_exc())
                return PlainTextResponse(message, status_code=500)

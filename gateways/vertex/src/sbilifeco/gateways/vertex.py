from __future__ import annotations
from io import BufferedIOBase, RawIOBase, TextIOBase
from typing import AsyncGenerator, AsyncIterator
from traceback import format_exc
from sbilifeco.boundaries.llm import ILLM
from sbilifeco.models.base import Response
from anthropic.lib.vertex import AsyncAnthropicVertex
from anthropic.lib.streaming import AsyncMessageStream
from sbilifeco.boundaries.material_reader import BaseMaterialReader


class VertexAI(ILLM, BaseMaterialReader):
    def __init__(self) -> None:
        self.region: str = ""
        self.project_id: str = ""
        self.model: str = ""
        self.max_output_tokens = 8192
        self.streams: dict[str, AsyncMessageStream] = {}

    def set_region(self, region: str) -> VertexAI:
        self.region = region
        return self

    def set_project_id(self, project_id: str) -> VertexAI:
        self.project_id = project_id
        return self

    def set_model(self, model: str) -> VertexAI:
        self.model = model
        return self

    def set_max_output_tokens(self, max_output_tokens: int) -> VertexAI:
        self.max_output_tokens = max_output_tokens
        return self

    async def async_init(self) -> None: ...

    async def async_shutdown(self) -> None: ...

    async def generate_reply(self, context: str) -> Response[str]:
        vertex_client: AsyncAnthropicVertex | None = None

        try:
            print(
                f"Initializing Vertex AI client with region: {self.region}, project_id: {self.project_id}, model: {self.model}",
                flush=True,
            )
            vertex_client = AsyncAnthropicVertex(
                region=self.region, project_id=self.project_id
            )

            message = await vertex_client.messages.create(
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": context,
                    }
                ],
                model=self.model,
                temperature=0,
            )

            return Response.ok(
                "\n".join(
                    [block.text for block in message.content if block.type == "text"]
                )
            )
        except Exception as e:
            print(f"Error generating reply with Vertex AI: {e}", flush=True)
            return Response.error(e)
        finally:
            print("Closing Vertex AI client", flush=True)
            if vertex_client:
                await vertex_client.close()

    async def generate_streamed_reply(
        self, request_id: str, context: str
    ) -> Response[AsyncGenerator[str, None]]:
        vertex_client: AsyncAnthropicVertex | None = None

        try:
            print(
                f"Initializing Vertex AI client with region: {self.region}, project_id: {self.project_id}, model: {self.model} for request_id: {request_id}",
                flush=True,
            )
            vertex_client = AsyncAnthropicVertex(
                region=self.region, project_id=self.project_id
            )

            print(
                f"Sending prompt and obtaining streamed response from Vertex AI for request_id: {request_id}",
                flush=True,
            )
            reply = vertex_client.messages.stream(
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": context,
                    }
                ],
                model=self.model,
                temperature=0,
            )

            stream = await reply.__aenter__()
            print(
                f"Stream obtained, processing streamed response for request_id: {request_id}",
                flush=True,
            )

            async def process_stream(request_id: str) -> AsyncGenerator[str, None]:
                stream = self.streams[request_id]

                try:
                    async for text in stream.text_stream:
                        yield text
                except Exception as e:
                    print(
                        f"Vertex Gateway: Error processing Vertex AI stream for request_id: {request_id}: {e}",
                        flush=True,
                    )
                    print(format_exc(), flush=True)
                    raise e
                finally:
                    print(
                        f"Vertex Gateway: Stream is exhausted, closing the open stream for request_id: {request_id}",
                        flush=True,
                    )
                    await stream.__aexit__(None, None, None)
                    print(
                        f"Vertex Gateway: Closing Vertex AI client for request_id: {request_id}",
                        flush=True,
                    )
                    await vertex_client.close()
                    del self.streams[request_id]

            self.streams[request_id] = stream
            return Response.ok(process_stream(request_id))
        except Exception as e:
            print(
                f"Vertex Gateway: Error generating streamed reply with Vertex AI for request_id: {request_id}: {e}",
                flush=True,
            )
            print(format_exc(), flush=True)
            return Response.error(e)
        finally:
            ...

    async def read_material(
        self,
        material: str | bytes | bytearray | RawIOBase | BufferedIOBase | TextIOBase,
    ) -> Response[str]:
        return await super().read_material(material)

    async def read_next_chunk(
        self, material_id: str
    ) -> Response[str | bytes | bytearray]:
        return await super().read_next_chunk(material_id)

from __future__ import annotations
from io import BufferedIOBase, RawIOBase, TextIOBase
from typing import AsyncGenerator, AsyncIterator
from traceback import format_exc
from anthropic.types import DocumentBlockParam
from sbilifeco.boundaries.llm import ILLM, LLMRequest
from sbilifeco.models.base import Response
from anthropic.lib.vertex import AsyncAnthropicVertex
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types.plain_text_source_param import PlainTextSourceParam
from anthropic.types.base64_pdf_source_param import Base64PDFSourceParam
from sbilifeco.boundaries.material_reader import BaseMaterialReader
from requests import Request, Session
from asyncio import get_running_loop
from functools import partial
from base64 import b64encode


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
        self, request: LLMRequest
    ) -> Response[AsyncGenerator[str, None]]:
        vertex_client: AsyncAnthropicVertex | None = None

        try:
            print(
                f"Initializing Vertex AI client with region: {self.region}, project_id: {self.project_id}, model: {self.model} for request_id: {request.request_id}",
                flush=True,
            )
            vertex_client = AsyncAnthropicVertex(
                region=self.region, project_id=self.project_id
            )

            print(
                f"Sending prompt and obtaining streamed response from Vertex AI for request_id: {request.request_id}",
                flush=True,
            )
            reply = vertex_client.messages.stream(
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": request.context,
                    }
                ],
                model=self.model,
                temperature=request.randomness,
            )

            stream = await reply.__aenter__()
            print(
                f"Stream obtained, processing streamed response for request_id: {request.request_id}",
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
                    del self.streams[request.request_id]

            self.streams[request.request_id] = stream
            return Response.ok(process_stream(request.request_id))
        except Exception as e:
            print(
                f"Vertex Gateway: Error generating streamed reply with Vertex AI for request_id: {request.request_id}: {e}",
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

    async def read_and_chunk(
        self,
        material: str | bytes | bytearray | RawIOBase | BufferedIOBase | TextIOBase,
    ) -> Response[AsyncIterator[str | bytes]]:
        try:
            source: bytes | str | None = None
            session: Session | None = None

            if isinstance(material, (bytes, bytearray)):
                source = bytes(material)
            elif isinstance(material, str):
                if material.startswith("file://"):
                    file_path = material[len("file://") :]
                    with open(file_path, "rb") as f:
                        source = f.read()
                elif material.startswith("http://") or material.startswith("https://"):
                    req = Request("GET", material).prepare()
                    session = Session()
                    http_response = await get_running_loop().run_in_executor(
                        None, partial(session.send, req)
                    )
                    source = http_response.content
                else:
                    source = material
            elif isinstance(material, (RawIOBase, BufferedIOBase)):
                source = material.read()
            elif isinstance(material, TextIOBase):
                source = material.read()

            if source is None:
                return Response.fail("Material is not in a supported source structure")

            source_as_block: PlainTextSourceParam | Base64PDFSourceParam
            if isinstance(source, str):
                source_as_block = {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": source,
                }
            else:
                base64_as_bytes = b64encode(source)
                source_as_block = {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_as_bytes.decode("utf-8"),
                }

            vertex_client = AsyncAnthropicVertex(
                region=self.region, project_id=self.project_id
            )

            reply = vertex_client.messages.stream(
                max_tokens=self.max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": ""
                        "You are a document parser. "
                        "Please parse the following document and extract it."
                        "If the content is base64 encoded, please decode it. Otherwise use it as it is."
                        "Split the document into logical chunks such as related paragraphs, bullet or numbered points, tables and figures."
                        'Flatten all tables as "Row header, Column header: Value".'
                        "Terms and conditions should appear in the same chunk as the original content on which they apply."
                        "Use the delimiter #=====# as the seperator between chunks."
                        "Return each logical chunk seperately",
                    },
                    {
                        "role": "user",
                        "content": [{"type": "document", "source": source_as_block}],
                    },
                ],
                model=self.model,
                temperature=0,
            )

            async def __stream() -> AsyncGenerator[str | bytes, None]:
                try:
                    async with reply as stream:
                        async for chunk in stream.text_stream:
                            yield chunk
                except Exception as e:
                    print(f"Error using Vertex AI client for chunking: {e}")
                    print(format_exc())
                finally:
                    print("Closing Vertex AI client in read_and_chunk", flush=True)
                    if vertex_client:
                        await vertex_client.close()
                    if session:
                        session.close()

            return Response.ok(__stream())

        except Exception as e:
            print(f"Error: {e}")
            print(format_exc())
            return Response.error(e)
        finally:
            ...

from __future__ import annotations
from typing import AsyncGenerator, Iterator
from io import BufferedIOBase, RawIOBase, TextIOBase
from uuid import uuid4
from magic import from_buffer
from sbilifeco.boundaries.llm import ILLM
from sbilifeco.boundaries.material_reader import (
    BaseMaterialReader,
    IMaterialReaderListener,
)
from sbilifeco.models.base import Response
from google import genai
from google.genai import types, Client as VertexClient
from google.genai.types import Part, GenerateContentResponse
import traceback


class VertexGemini(ILLM, BaseMaterialReader):
    def __init__(self) -> None:
        BaseMaterialReader.__init__(self)
        self.vertex_client: VertexClient
        self.region: str = ""
        self.project_id: str = ""
        self.model: str = ""
        self.max_output_tokens = 8192
        self.streams: dict[str, AsyncGenerator] = {}

    def set_region(self, region: str) -> VertexGemini:
        self.region = region
        return self

    def set_project_id(self, project_id: str) -> VertexGemini:
        self.project_id = project_id
        return self

    def set_model(self, model: str) -> VertexGemini:
        self.model = model
        return self

    def set_max_output_tokens(self, max_output_tokens: int) -> VertexGemini:
        self.max_output_tokens = max_output_tokens
        return self

    async def async_init(self) -> None:
        self.vertex_client = VertexClient(
            vertexai=True, location=self.region, project=self.project_id
        )

    async def async_shutdown(self) -> None:
        self.vertex_client.close()

    async def generate_reply(self, context: str) -> Response[str]:
        try:
            llm_response = self.vertex_client.models.generate_content(
                model=self.model,
                contents=context,
                config=types.GenerateContentConfig(temperature=0.0),
            )

            return Response.ok(llm_response.text)
        except Exception as e:
            traceback.print_exc()
            return Response.error(e)

    async def read_material(
        self,
        material: str | bytes | bytearray | RawIOBase | BufferedIOBase | TextIOBase,
    ) -> Response[str]:
        try:
            material_id = uuid4().hex

            material_as_bytes: bytes | None = None
            referred_mime: str | None = None

            if isinstance(material, bytearray):
                material_as_bytes = bytes(material)
            elif isinstance(material, bytes):
                material_as_bytes = material
            elif isinstance(material, str):
                if material.lower().startswith("file://"):
                    file_path = material[7:]
                    with open(file_path, "rb") as f:
                        material_as_bytes = f.read()
                else:
                    material_as_bytes = material.encode("utf-8")
            elif isinstance(material, (RawIOBase, BufferedIOBase)):
                material_as_bytes = material.read()
            elif isinstance(material, TextIOBase):
                material_as_bytes = material.read().encode("utf-8")

            if material_as_bytes is None:
                return Response.fail("Unsupported sourcer material provided.", 400)

            referred_mime = from_buffer(material_as_bytes, mime=True)

            llm_result = self.vertex_client.models.generate_content_stream(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=material_as_bytes,
                        mime_type=referred_mime or "application/pdf",
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            self.streams[material_id] = self._fetch_next_chunk(llm_result)
            return Response.ok(material_id)
        except Exception as e:
            return Response.error(e)

    async def read_next_chunk(
        self, material_id: str
    ) -> Response[str | bytes | bytearray]:
        try:
            chunk_source = self.streams.get(material_id)
            if chunk_source is None:
                return Response.fail(
                    f"Unable to find chunked material {material_id}", 404
                )

            chunk = await anext(chunk_source)
            return Response.ok(chunk)
        except Exception as e:
            return Response.error(e)

    async def _fetch_next_chunk(
        self, chunks_by_llm: Iterator[GenerateContentResponse]
    ) -> AsyncGenerator[str | None, None]:
        for chunk in chunks_by_llm:
            yield chunk.text

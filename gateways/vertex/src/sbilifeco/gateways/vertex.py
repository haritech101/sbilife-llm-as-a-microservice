from __future__ import annotations
from io import BufferedIOBase, RawIOBase, TextIOBase
from sbilifeco.boundaries.llm import ILLM
from sbilifeco.models.base import Response
from anthropic import AsyncAnthropicVertex
from sbilifeco.boundaries.material_reader import BaseMaterialReader


class VertexAI(ILLM, BaseMaterialReader):
    def __init__(self) -> None:
        self.region: str = ""
        self.project_id: str = ""
        self.model: str = ""
        self.max_output_tokens = 8192

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

    async def read_material(
        self,
        material: str | bytes | bytearray | RawIOBase | BufferedIOBase | TextIOBase,
    ) -> Response[str]:
        return await super().read_material(material)

    async def read_next_chunk(
        self, material_id: str
    ) -> Response[str | bytes | bytearray]:
        return await super().read_next_chunk(material_id)

from asyncio import run, sleep
from os import getenv
from re import M
from typing import NoReturn

from dotenv import load_dotenv
from sbilifeco.cp.llm.http_server import LLMHttpServer
from sbilifeco.cp.material_reader.http_server import MaterialReaderHttpServer
from sbilifeco.gateways.vertex_gemini import VertexGemini

from envvars import Defaults, EnvVars


class VertexLLMMicroservice:
    async def start(self):
        # Settings from environment
        region = getenv(EnvVars.vertex_ai_region, Defaults.vertex_ai_region)
        project_id = getenv(EnvVars.vertex_ai_project_id, "")
        model = getenv(EnvVars.vertex_ai_model, Defaults.vertex_ai_model)
        http_port_qa = int(getenv(EnvVars.http_port_qa, Defaults.http_port_qa))
        http_port_material = int(
            getenv(EnvVars.http_port_material, Defaults.http_port_material)
        )
        min_chunk_size = int(getenv(EnvVars.min_chunk_size, Defaults.min_chunk_size))

        # Vertex gateway
        self.vertex = (
            VertexGemini()
            .set_region(region)
            .set_project_id(project_id)
            .set_model(model)
            .set_min_chunk_size(min_chunk_size)
        )
        await self.vertex.async_init()

        # HTTP server
        self.http_server_qa = LLMHttpServer()
        self.http_server_qa.set_llm(self.vertex).set_http_port(http_port_qa)
        await self.http_server_qa.listen()

        self.http_server_material = MaterialReaderHttpServer()
        self.http_server_material.set_material_reader(self.vertex).set_http_port(
            http_port_material
        )
        await self.http_server_material.listen()

    async def run_forever(self) -> NoReturn:
        await self.start()
        while True:
            await sleep(3600)


if __name__ == "__main__":
    load_dotenv()
    run(VertexLLMMicroservice().run_forever())

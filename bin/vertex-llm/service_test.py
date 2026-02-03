import sys

sys.path.append("./src")

from os import getenv
from unittest import IsolatedAsyncioTestCase
from dotenv import load_dotenv
from envvars import EnvVars, Defaults

# Import the necessary service(s) here
from service import VertexLLMMicroservice
from sbilifeco.cp.llm.http_client import LLMHttpClient
from sbilifeco.cp.material_reader.http_client import MaterialReaderHttpClient


class Test(IsolatedAsyncioTestCase):
    BROCHURE_PATH = "fixtures/brochure.pdf"

    async def asyncSetUp(self) -> None:
        load_dotenv()

        self.test_type = getenv(EnvVars.test_type, Defaults.test_type)
        http_port_qa = int(getenv(EnvVars.http_port_qa, Defaults.http_port_qa))
        http_port_material = int(
            getenv(EnvVars.http_port_material, Defaults.http_port_material)
        )
        self.min_chunk_size = int(
            getenv(EnvVars.min_chunk_size, Defaults.min_chunk_size)
        )

        # Initialise the service(s) here
        if self.test_type == "unit":
            self.service = VertexLLMMicroservice()
            await self.service.start()

        # Initialise the client(s) here
        self.llm_client = LLMHttpClient()
        self.llm_client.set_proto("http").set_host("localhost").set_port(http_port_qa)

        # Material reader client
        self.material_reader_client = MaterialReaderHttpClient()
        self.material_reader_client.set_proto("http").set_host("localhost").set_port(
            http_port_material
        )

    async def asyncTearDown(self) -> None:
        # Shutdown the service(s) here
        # await self.service.async_shutdown()
        ...

    async def test(self) -> None:
        # Arrange
        question = "What is the meaning of life, the universe, and everything?"

        # Act
        response = await self.llm_client.generate_reply(question)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None

        self.assertIn("42", response.payload)

    async def test_read_and_chunk(self) -> None:
        # Arrange
        with open(self.BROCHURE_PATH, "rb") as brochure:
            brochure_bytes = brochure.read()

        # Act
        response = await self.material_reader_client.read_material(brochure_bytes)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None
        self.assertTrue(response.payload)

        material_id = response.payload

        # Act
        chunk_response = await self.material_reader_client.read_next_chunk(material_id)

        # Assert
        self.assertTrue(chunk_response.is_success, chunk_response.message)
        assert chunk_response.payload is not None
        self.assertTrue(chunk_response.payload)
        self.assertGreater(len(chunk_response.payload), self.min_chunk_size)

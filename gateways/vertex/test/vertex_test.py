import sys
from uuid import uuid4

sys.path.append("./src")

from os import getenv
from unittest import IsolatedAsyncioTestCase
from dotenv import load_dotenv
from envvars import EnvVars, Defaults

# Import the necessary service(s) here
from sbilifeco.boundaries.llm import LLMRequest
from sbilifeco.gateways.vertex import VertexAI
from sbilifeco.gateways.vertex_gemini import VertexGemini


class Test(IsolatedAsyncioTestCase):
    BROCHURE_PATH = "test/fixtures/brochure.pdf"

    async def asyncSetUp(self) -> None:
        load_dotenv()

        region = getenv(EnvVars.vertex_ai_region, Defaults.vertex_ai_region)
        project_id = getenv(EnvVars.vertex_ai_project_id, "")
        model = getenv(EnvVars.vertex_ai_model, Defaults.vertex_ai_model)
        self.min_chunk_size = int(
            getenv(EnvVars.min_chunk_size, Defaults.min_chunk_size)
        )

        # Initialise the service(s) here
        self.claude_service = (
            VertexAI().set_region(region).set_project_id(project_id).set_model(model)
        )
        await self.claude_service.async_init()

        self.gemini_service = (
            VertexGemini()
            .set_region(region)
            .set_project_id(project_id)
            .set_model(model)
        )

        await self.gemini_service.async_init()

    async def asyncTearDown(self) -> None:
        # Shutdown the service(s) here
        await self.gemini_service.async_shutdown()
        await self.claude_service.async_shutdown()

    async def test_claude(self) -> None:
        response = await self.claude_service.generate_reply(
            "What is the answer to life, the universe and everything?"
        )
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None

        print(response.payload, flush=True)

    async def test_gemini(self) -> None:
        response = await self.gemini_service.generate_reply(
            "What is the answer to life, the universe and everything?"
        )
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None

        print(response.payload, flush=True)

    async def test_streaming(self) -> None:
        # Arrange
        request = LLMRequest(
            context="What is the answer to life, the universe and everything?"
        )

        # Act
        response = await self.claude_service.generate_streamed_reply(request)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None

        async for chunk in response.payload:
            print(chunk, flush=True)

    async def test_read_and_chunk(self) -> None:
        # Arrange
        with open(self.BROCHURE_PATH, "rb") as brochure:
            brochure_bytes = brochure.read()

        # Act
        response = await self.gemini_service.read_material(brochure_bytes)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None
        self.assertTrue(response.payload)

        material_id = response.payload

        # Act
        chunk_response = await self.gemini_service.read_next_chunk(material_id)

        # Assert
        self.assertTrue(chunk_response.is_success, chunk_response.message)
        assert chunk_response.payload is not None
        self.assertTrue(chunk_response.payload)
        self.assertGreaterEqual(len(chunk_response.payload), self.min_chunk_size)

    async def test_read_and_chunk_iterator(self) -> None:
        # Arrange
        raw_input = "file://./test/fixtures/brochure.pdf"

        # Act
        read_and_chunk_response = await self.claude_service.read_and_chunk(raw_input)

        # Assert
        self.assertTrue(
            read_and_chunk_response.is_success, read_and_chunk_response.message
        )

        stream = read_and_chunk_response.payload
        assert stream is not None

        async for chunk in stream:
            self.assertTrue(chunk)
            print(chunk, end="", flush=True)

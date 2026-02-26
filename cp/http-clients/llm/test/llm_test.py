import sys
from typing import AsyncGenerator
from uuid import uuid4

sys.path.append("./src")

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from faker import Faker
from sbilifeco.models.base import Response
from sbilifeco.boundaries.llm import ILLM
from sbilifeco.cp.llm.http_server import LLMHttpServer
from sbilifeco.cp.llm.http_client import LLMHttpClient
from random import randint


class LLMTest(IsolatedAsyncioTestCase):
    HTTP_PORT = 8181

    async def asyncSetUp(self) -> None:
        self.llm: ILLM = AsyncMock(spec=ILLM)

        self.http_server = LLMHttpServer()
        self.http_server.set_llm(self.llm).set_http_port(self.HTTP_PORT)
        await self.http_server.listen()

        self.client = LLMHttpClient()
        self.client.set_proto("http").set_host("localhost").set_port(self.HTTP_PORT)

        self.faker = Faker()

    async def asyncTearDown(self) -> None:
        await self.http_server.stop()
        patch.stopall()
        return await super().asyncTearDown()

    async def test_sequence(self) -> None:
        # Arrange
        question = self.faker.sentence()
        reply = self.faker.paragraph()
        patched_generate_reply = patch.object(
            self.llm, "generate_reply", return_value=Response.ok(reply)
        ).start()

        # Act
        response = await self.client.generate_reply(question)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None
        patched_generate_reply.assert_called_once_with(question)

    async def test_series(self) -> None:
        # Arrange
        question = "What is the meaning of life, the universe, and everything?"
        request_id = uuid4().hex
        fn_stream = patch.object(
            self.llm,
            "generate_streamed_reply",
            return_value=Response.ok(self.__generate_stream()),
        ).start()

        # Act
        response = await self.client.generate_streamed_reply(request_id, question)

        # Assert
        fn_stream.assert_called_once_with(request_id, question)

        # Assert
        self.assertTrue(response.is_success, response.message)
        assert response.payload is not None

        chunk = await response.payload.__anext__()
        self.assertIsInstance(chunk, str)
        print(f"Received first chunk: {chunk}")

        async for chunk in response.payload:
            self.assertIsInstance(chunk, str)
            print(f"Received chunk: {chunk}")

    async def __generate_stream(self) -> AsyncGenerator[str, None]:
        for _ in range(randint(1, 5)):
            yield self.faker.paragraph()

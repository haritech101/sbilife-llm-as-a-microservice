from typing import AsyncGenerator
from traceback import format_exc

from asyncio import get_running_loop
from functools import partial
from sbilifeco.cp.common.http.client import HttpClient
from sbilifeco.boundaries.llm import ILLM, ChatMessage
from sbilifeco.models.base import Response
from sbilifeco.cp.llm.paths import Paths, LLMQuery
from requests import Request, Session


class LLMHttpClient(HttpClient, ILLM):
    async def generate_reply(self, context: str) -> Response[str]:
        try:
            return await self.request_as_model(
                Request(
                    method="POST",
                    url=f"{self.url_base}{Paths.QUERIES}",
                    json=LLMQuery(context=context).model_dump(),
                )
            )
        except Exception as e:
            return Response.error(e)

    async def generate_streamed_reply(
        self, request_id: str, context: str
    ) -> Response[AsyncGenerator[str, None]]:
        try:
            req = Request(
                method="POST",
                url=f"{self.url_base}{Paths.STREAMS.format(request_id=request_id)}",
                json=LLMQuery(context=context).model_dump(),
            )
            with Session() as session:
                prepped = session.prepare_request(req)
                http_response = await get_running_loop().run_in_executor(
                    None, partial(session.send, prepped, stream=True)
                )

                async def stream_generator():
                    try:
                        for content in http_response.iter_content(
                            4096, decode_unicode=True
                        ):
                            yield content
                    except Exception as e:
                        print(f"Error in stream_generator: {e}")
                        print(format_exc())
                        return

            return Response.ok(stream_generator())
        except Exception as e:
            print(f"Error in generate_streamed_reply: {e}")
            print(format_exc())
            return Response.error(e)

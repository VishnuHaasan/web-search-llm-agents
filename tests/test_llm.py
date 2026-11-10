import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm.client import LLMClient, LLMError

KEY = "sk-secret-123"


def reply(content="  The answer.  ", finish="stop", **message):
    return {"choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish}]}


def llm(handler, timeout=5.0, temperature=0.2):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://gw.test/v1", transport=transport,
                               headers={"Authorization": f"Bearer {KEY}"})
    return LLMClient(client=client, model="m-1", timeout=timeout, temperature=temperature)


def ok(body=None):
    return lambda request: httpx.Response(200, json=body or reply())


# --- request -------------------------------------------------------------------------

async def test_request_shape():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json=reply())

    await llm(handler, temperature=0.3).chat("SYS", "USER", 123)
    assert seen["url"] == "http://gw.test/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {"model": "m-1", "max_tokens": 123, "temperature": 0.3,
                            "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]}


# --- success -------------------------------------------------------------------------

async def test_returns_stripped_content():
    assert await llm(ok()).chat("s", "u", 10) == "The answer."


async def test_reasoning_content_ignored():
    out = await llm(ok(reply("Final.", reasoning_content="thinking..."))).chat("s", "u", 10)
    assert out == "Final."


# --- failures ------------------------------------------------------------------------

@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
async def test_bad_status(status):
    with pytest.raises(LLMError, match=str(status)):
        await llm(lambda r: httpx.Response(status, text=f"error, your key {KEY}")).chat("s", "u", 10)


async def test_timeout_is_a_hard_cap():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=reply())

    with pytest.raises(LLMError, match="(?i)timeout"):
        await llm(slow, timeout=0.1).chat("s", "u", 10)


async def test_connection_error():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="ConnectError"):
        await llm(down).chat("s", "u", 10)


@pytest.mark.parametrize("body", [
    b"not json",
    b"{}",
    b'{"choices": []}',
    b'{"choices": [{}]}',
    b'{"choices": [{"message": null}]}',
    b"[1, 2]",
])
async def test_malformed_body(body):
    with pytest.raises(LLMError, match="(?i)bad response"):
        await llm(lambda r: httpx.Response(200, content=body)).chat("s", "u", 10)


@pytest.mark.parametrize("content", [None, "", "   \n "])
async def test_empty_answer(content):
    with pytest.raises(LLMError, match="(?i)empty"):
        await llm(ok(reply(content, reasoning_content="ran out while thinking"))).chat("s", "u", 10)


async def test_truncated_answer():
    with pytest.raises(LLMError, match="(?i)truncated"):
        await llm(ok(reply("half an ans", finish="length"))).chat("s", "u", 10)


async def test_key_never_in_error_messages():
    for handler in (lambda r: httpx.Response(401, text=f"bad key {KEY}"),
                    lambda r: httpx.Response(200, content=f"oops {KEY}".encode())):
        with pytest.raises(LLMError) as e:
            await llm(handler).chat("s", "u", 10)
        assert KEY not in str(e.value)


# --- lifespan ------------------------------------------------------------------------

def test_no_llm_url_means_no_client(monkeypatch, filters_dir):
    from api.app import app
    from search.filters import Filters
    monkeypatch.setenv("LLM_URL", "")                 # load_dotenv does not override an existing variable
    try:
        with TestClient(app):
            assert app.state.llm_client is None
    finally:
        Filters.load(filters_dir)


def test_llm_url_creates_client(monkeypatch, filters_dir):
    from api.app import app
    from search.filters import Filters
    monkeypatch.setenv("LLM_URL", "http://gw.test/v1")
    monkeypatch.setenv("LLM_MODEL", "m-1")
    monkeypatch.setenv("LLM_TIMEOUT", "17")
    try:
        with TestClient(app):
            c = app.state.llm_client
            assert isinstance(c, LLMClient) and c.model == "m-1" and c.timeout == 17
            assert c.client.timeout.read >= 17                # the httpx timeout must not cut the call before LLM_TIMEOUT
    finally:
        Filters.load(filters_dir)



# --- image content and model override (vision descriptions) --------------------------

async def test_list_content_and_model_override():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply())

    parts = [{"type": "text", "text": "Describe this image."},
             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]
    await llm(handler).chat("SYS", parts, 100, model="vision-1")
    assert seen["body"]["model"] == "vision-1"
    assert seen["body"]["messages"][1] == {"role": "user", "content": parts}


async def test_model_defaults_to_the_client_model():
    seen = {}

    def handler(request):
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=reply())

    await llm(handler).chat("s", "u", 10)
    assert seen["model"] == "m-1"

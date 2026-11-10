import asyncio
import base64

import httpx
import pytest

import llm.images as images
from fetch.safe_fetch import IMAGE_CONTENT_TYPES, IMAGE_MAX_BYTES, InvalidContentType, BlockedURL
from llm.client import LLMError
from llm.images import describe_image, describe_images
from search.schema import ImageCandidate


def img(n=0, thumbnail="auto"):
    return ImageCandidate(url=f"https://full.com/{n}.jpg", provider="4get",
                          thumbnail=f"https://thumb.com/{n}.jpg" if thumbnail == "auto" else thumbnail)


class FakeLLM:
    def __init__(self, reply="A blue and yellow logo.", error=None, delay=0.0):
        self.reply, self.error, self.delay, self.calls = reply, error, delay, []

    async def chat(self, system, user, max_tokens, model=None):
        self.calls.append((system, user, max_tokens, model))
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def fetched(monkeypatch):
    """Replace safe_get; records (url, types, max_bytes). Set .error / .delay / .ctype to change behaviour."""
    class Fetch:
        calls, error, delay, ctype = [], None, 0.0, "image/png"

        async def __call__(self, client, url, types, max_bytes):
            self.calls.append((url, types, max_bytes))
            await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return url, b"\x89PNGbytes", self.ctype

    f = Fetch()
    f.calls = []
    monkeypatch.setattr(images, "safe_get", f)
    return f


# --- describe_image ----------------------------------------------------------------------

async def test_thumbnail_fetched_in_image_mode(fetched):
    llm = FakeLLM()
    assert await describe_image(llm, None, img(), "vision-1", 2) == "A blue and yellow logo."
    assert fetched.calls == [("https://thumb.com/0.jpg", IMAGE_CONTENT_TYPES, IMAGE_MAX_BYTES)]


async def test_full_url_when_no_thumbnail(fetched):
    await describe_image(FakeLLM(), None, img(thumbnail=None), "vision-1", 2)
    assert fetched.calls[0][0] == "https://full.com/0.jpg"


async def test_request_has_data_url_with_real_content_type(fetched):
    llm = FakeLLM()
    await describe_image(llm, None, img(), "vision-1", 2)
    [(system, user, max_tokens, model)] = llm.calls
    assert model == "vision-1" and max_tokens == 100 and system.startswith("Describe the image")
    text, image = user
    assert text == {"type": "text", "text": "Describe this image."}
    assert image["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNGbytes").decode()


@pytest.mark.parametrize("error", [InvalidContentType("text/html"), BlockedURL("private"),
                                   httpx.ConnectError("refused")])
async def test_fetch_failure_gives_none(fetched, error):
    fetched.error = error
    llm = FakeLLM()
    assert await describe_image(llm, None, img(), "vision-1", 2) is None and llm.calls == []


async def test_llm_error_gives_none(fetched):
    assert await describe_image(FakeLLM(error=LLMError("HTTP 400")), None, img(), "vision-1", 2) is None


@pytest.mark.parametrize("slow", ["fetch", "llm"])
async def test_timeout_covers_fetch_and_llm(fetched, slow):
    if slow == "fetch":
        fetched.delay = 5
    llm = FakeLLM(delay=5 if slow == "llm" else 0)
    assert await describe_image(llm, None, img(), "vision-1", 0.1) is None


async def test_other_errors_are_not_hidden(fetched):
    with pytest.raises(RuntimeError):
        await describe_image(FakeLLM(error=RuntimeError("bug")), None, img(), "vision-1", 2)


# --- describe_images ---------------------------------------------------------------------

@pytest.mark.parametrize("n, limit", [(10, 10), (3, 10), (10, 4), (0, 10)])
async def test_one_entry_per_image(fetched, n, limit):
    out = await describe_images(FakeLLM(), None, [img(i) for i in range(n)], "vision-1", limit, 2)
    assert len(out) == n
    assert out == ["A blue and yellow logo."] * min(n, limit) + [None] * max(0, n - limit)


async def test_order_kept_and_failures_are_none(monkeypatch):
    async def fetch(client, url, types, max_bytes):
        if url.endswith("/1.jpg"):
            raise InvalidContentType("text/html")
        await asyncio.sleep(0.05 if url.endswith("/0.jpg") else 0)       # the first one finishes last
        return url, url.encode(), "image/jpeg"

    class Echo(FakeLLM):
        async def chat(self, system, user, max_tokens, model=None):
            return base64.b64decode(user[1]["image_url"]["url"].split(",", 1)[1]).decode()

    monkeypatch.setattr(images, "safe_get", fetch)
    out = await describe_images(Echo(), None, [img(i) for i in range(3)], "vision-1", 10, 2)
    assert out == ["https://thumb.com/0.jpg", None, "https://thumb.com/2.jpg"]


async def test_images_run_together(fetched):
    llm = FakeLLM(delay=0.3)
    t = asyncio.get_running_loop().time()
    await describe_images(llm, None, [img(i) for i in range(5)], "vision-1", 10, 2)
    assert asyncio.get_running_loop().time() - t < 0.6

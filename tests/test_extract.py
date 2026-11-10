import asyncio
import time

import httpx
import pytest

import fetch.safe_fetch as sf
from fetch.extract import Extractor, cap, default_favicon, find_favicon

ARTICLE = ("<html><head><title>T</title></head><body><article>"
           + "<p>Asyncio task groups run coroutines together and cancel the rest when one fails. </p>" * 2
           + "<p>They replace gather in most new code, because errors are never silently lost. </p>" * 2
           + "</article></body></html>")


class SlowStream(httpx.AsyncByteStream):
    """A body that trickles out, so a cancel lands mid-download."""
    closed = 0

    async def __aiter__(self):
        for _ in range(100):
            await asyncio.sleep(0.05)
            yield b"<p>x</p>"

    async def aclose(self):
        SlowStream.closed += 1


async def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/slow"):
        await asyncio.sleep(float(path.removeprefix("/slow")))
    if path == "/trickle":
        return httpx.Response(200, headers={"Content-Type": "text/html"}, stream=SlowStream())
    if path == "/icon":
        page = ARTICLE.replace("<title>T</title>", '<title>T</title><link rel="shortcut icon" href="/static/i.png">')
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=page.replace("Asyncio task", "Asyncio <b>task</b>"))
    if path == "/plain":
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="line\n" * 100)
    if path == "/empty":
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text="<html><body><nav>menu</nav></body></html>")
    if path == "/missing":
        return httpx.Response(404, headers={"Content-Type": "text/html"}, text="gone")
    return httpx.Response(200, headers={"Content-Type": "text/html"}, text=ARTICLE)


@pytest.fixture
def extractor(monkeypatch):
    async def public(host, port):
        return "93.184.216.34"
    monkeypatch.setattr(sf, "resolve_public", public)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    return Extractor(client=client, page_timeout=8, min_text_filter=50, concurrency=8)


async def test_returns_what_finished(extractor):
    out = await extractor.extract_many(["https://a.com/fast", "https://a.com/slow5"], 0.5)
    assert out[0].text and out[0].error is None
    assert out[1].text is None and "budget" in out[1].error


async def test_order_kept_when_later_pages_finish_first(extractor):
    urls = ["https://a.com/slow0.3", "https://b.com/fast", "https://c.com/slow0.1"]
    out = await extractor.extract_many(urls, 2)
    assert [e.url for e in out] == urls and all(e.text for e in out)


async def test_stops_at_the_deadline(extractor):
    t = time.perf_counter()
    await extractor.extract_many([f"https://s{n}.com/slow5" for n in range(20)], 0.3)
    assert time.perf_counter() - t < 1


async def test_cancelled_download_closes_its_stream(extractor):
    SlowStream.closed = 0
    out = await extractor.extract_many(["https://a.com/trickle"], 0.2)
    assert "budget" in out[0].error and SlowStream.closed == 1


async def test_no_urls(extractor):
    assert await extractor.extract_many([], 1) == []


# --- step 6c: long-page cap ------------------------------------------------------------

def test_cap_under_limit_unchanged():
    assert cap("short", 10) == "short" and cap(None, 10) is None


def test_cap_cuts_at_last_line_break():
    assert cap("aaaa\nbbbb\ncccc", 12) == "aaaa\nbbbb"


def test_cap_hard_cut_without_line_break():
    assert cap("x" * 50, 10) == "x" * 10


async def test_cap_applied_to_text_and_markdown(extractor):
    extractor.max_page_chars = 120
    [e] = await extractor.extract_many(["https://a.com/icon"], 2, markdown=True)
    assert len(e.text) <= 120 and len(e.markdown) <= 120
    [p] = await extractor.extract_many(["https://a.com/plain"], 2)
    assert p.text == "line\n" * 23 + "line"        # 119 chars, cut before the next line break


# --- step 6b: favicon --------------------------------------------------------------------

@pytest.mark.parametrize("head, expected", [
    ('<link rel="icon" href="/fav.png">', "https://a.com/fav.png"),
    ('<link rel="shortcut icon" href="../_static/py.svg">', "https://a.com/_static/py.svg"),
    ('<link rel="apple-touch-icon" href="https://cdn.x.com/i.png">', "https://cdn.x.com/i.png"),
    ('<link rel="icon" href="data:image/png;base64,AAAA"><link rel="icon" href="/real.ico">', "https://a.com/real.ico"),
    ('<link rel="icon" href="javascript:alert(1)">', "https://a.com/favicon.ico"),
    ('<link rel="stylesheet" href="/s.css">', "https://a.com/favicon.ico"),
    ("", "https://a.com/favicon.ico"),
])
def test_find_favicon(head, expected):
    body = f"<html><head>{head}</head><body>x</body></html>".encode()
    assert find_favicon(body, "https://a.com/3/page.html") == expected


def test_find_favicon_garbage_body_falls_back():
    assert find_favicon(b"", "https://a.com/x") == "https://a.com/favicon.ico"


@pytest.mark.parametrize("url, expected", [
    ("https://a.com/x/y?q=1", "https://a.com/favicon.ico"),
    ("http://a.com:8080/", "http://a.com:8080/favicon.ico"),
    ("ftp://a.com/", None),
    ("not a url", None),
])
def test_default_favicon(url, expected):
    assert default_favicon(url) == expected


async def test_favicon_from_page(extractor):
    [e] = await extractor.extract_many(["https://a.com/icon"], 2)
    assert e.favicon == "https://a.com/static/i.png"


@pytest.mark.parametrize("path", ["/plain", "/empty", "/missing"])
async def test_favicon_fallback_for_plain_and_failed_pages(extractor, path):
    [e] = await extractor.extract_many([f"https://a.com{path}"], 2)
    assert e.favicon == "https://a.com/favicon.ico"


async def test_favicon_for_budget_exceeded(extractor):
    [e] = await extractor.extract_many(["https://a.com/slow5"], 0.1)
    assert "budget" in e.error and e.favicon == "https://a.com/favicon.ico"


# --- step 6a: markdown ---------------------------------------------------------------------

async def test_markdown_only_when_asked(extractor):
    [plain] = await extractor.extract_many(["https://a.com/icon"], 2)
    [md] = await extractor.extract_many(["https://a.com/icon"], 2, markdown=True)
    assert plain.markdown is None
    assert "**task**" in md.markdown and "**" not in md.text


async def test_no_markdown_for_plain_text_or_failed_pages(extractor):
    out = await extractor.extract_many(["https://a.com/plain", "https://a.com/empty"], 2, markdown=True)
    assert [e.markdown for e in out] == [None, None]

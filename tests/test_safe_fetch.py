import asyncio
import gzip
import ipaddress
import socket

import httpx
import pytest

import fetch.safe_fetch as sf
from fetch.safe_fetch import (BlockedURL, BodyTooLarge, InvalidContentType, UnsuccessfulFetch, check_url,
                              make_client, resolve_public, safe_get)

HTML = {"Content-Type": "text/html; charset=utf-8"}

# fake DNS: host -> addresses. IP literals resolve to themselves, like the real getaddrinfo
DNS = {
    "public.com": ["93.184.216.34"],
    "other.com": ["151.101.1.69"],
    "v6.com": ["2606:4700::6810:84e5"],
    "internal.com": ["10.0.0.5"],
    "localhost": ["127.0.0.1", "::1"],
    "mixed.com": ["93.184.216.34", "192.168.1.10"],      # one private answer is enough to refuse
    "rebind.com": ["1.1.1.1", "10.0.0.5"],               # same, with the public address sorting first
    "metadata.com": ["169.254.169.254"],
}


@pytest.fixture(autouse=True)
async def fake_dns(monkeypatch):
    async def getaddrinfo(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host.split("%")[0])
            addrs = [host]
        except ValueError:
            if host not in DNS:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            addrs = DNS[host]
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                for a in addrs]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)


class Body(httpx.AsyncByteStream):
    """A response body that counts how much was read and whether it was closed."""

    def __init__(self, chunks):
        self.chunks, self.read, self.closed = chunks, 0, False

    async def __aiter__(self):
        for c in self.chunks:
            self.read += len(c)
            yield c

    async def aclose(self):
        self.closed = True


class Server:
    """MockTransport handler: routes by the Host header, records every request it receives."""

    def __init__(self, routes):
        self.routes = routes                    # (host, path) -> (status, headers, chunks)
        self.requests: list[httpx.Request] = []
        self.bodies: list[Body] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, headers, chunks = self.routes.get((request.headers["host"], request.url.path), (404, HTML, [b"nf"]))
        body = Body(chunks)
        self.bodies.append(body)
        return httpx.Response(status, headers=headers, stream=body)

    @property
    def hosts(self):
        return [r.headers["host"] for r in self.requests]


def client(server):
    return httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=False, trust_env=False)


def page(text=b"<p>hello</p>", headers=HTML):
    return 200, headers, [text]


def redirect(location, status=302):
    return status, {"Location": location}, [b""]


# --- check_url -------------------------------------------------------------------

@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://public.com/x", "javascript:alert(1)",
                                 "gopher://public.com/", "//public.com/x"])
def test_only_http_and_https(url):
    with pytest.raises(BlockedURL, match="scheme"):
        check_url(url)


def test_no_host():
    with pytest.raises(BlockedURL, match="no host"):
        check_url("http:///path")


@pytest.mark.parametrize("url", ["https://user:pw@public.com/", "https://user@public.com/"])
def test_credentials_refused(url):
    with pytest.raises(BlockedURL, match="credentials"):
        check_url(url)


@pytest.mark.parametrize("url", ["http://public.com:8080/", "https://public.com:22/", "http://public.com:6379/"])
def test_other_ports_refused(url):
    with pytest.raises(BlockedURL, match="port"):
        check_url(url)


@pytest.mark.parametrize("url, expected", [
    ("http://public.com/x", ("public.com", 80)),
    ("https://public.com/x", ("public.com", 443)),
    ("https://public.com:443/x", ("public.com", 443)),
    ("http://PUBLIC.com:80/", ("public.com", 80)),
])
def test_allowed(url, expected):
    assert check_url(url) == expected


# --- resolve_public --------------------------------------------------------------

@pytest.mark.parametrize("host", ["public.com", "v6.com", "93.184.216.34"])
async def test_public_hosts_resolve(host):
    ip = await resolve_public(host, 443)
    assert ipaddress.ip_address(ip).is_global


@pytest.mark.parametrize("host", [
    "localhost", "internal.com", "mixed.com", "rebind.com", "metadata.com",
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.0.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
    "::1", "fe80::1%en0", "fc00::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
])
async def test_non_public_refused(host):
    with pytest.raises(BlockedURL, match="non-public"):
        await resolve_public(host, 443)


async def test_unresolvable_host():
    with pytest.raises(BlockedURL, match="cannot resolve"):
        await resolve_public("no-such-host.invalid", 443)


# --- safe_get: connection ----------------------------------------------------------

async def test_connects_to_the_checked_ip_with_the_real_host_name():
    s = Server({("public.com", "/a"): page()})
    final_url, body, ctype = await safe_get(client(s), "https://public.com/a?q=1")
    assert (final_url, body, ctype) == ("https://public.com/a?q=1", b"<p>hello</p>", "text/html")
    req = s.requests[0]
    assert req.url.host == "93.184.216.34" and req.url.params["q"] == "1"
    assert req.headers["host"] == "public.com"
    assert req.extensions["sni_hostname"] == "public.com"      # TLS certificate is checked against the name


async def test_ipv6_address_is_bracketed():
    s = Server({("v6.com", "/"): page()})
    await safe_get(client(s), "https://v6.com/")
    assert s.requests[0].url.host == "2606:4700::6810:84e5"


async def test_blocked_url_makes_no_request():
    s = Server({})
    for url in ["http://localhost/", "http://internal.com/", "http://public.com:8080/", "file:///etc/passwd"]:
        with pytest.raises(BlockedURL):
            await safe_get(client(s), url)
    assert s.requests == []


# --- safe_get: redirects -------------------------------------------------------------

async def test_redirect_followed_and_final_url_returned():
    s = Server({("public.com", "/old"): redirect("https://other.com/new"), ("other.com", "/new"): page()})
    final_url, body, _ = await safe_get(client(s), "https://public.com/old")
    assert final_url == "https://other.com/new" and body == b"<p>hello</p>"
    assert s.hosts == ["public.com", "other.com"]


async def test_relative_redirect():
    s = Server({("public.com", "/a/b"): redirect("../c"), ("public.com", "/c"): page()})
    final_url, _, _ = await safe_get(client(s), "https://public.com/a/b")
    assert final_url == "https://public.com/c"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_every_redirect_status(status):
    s = Server({("public.com", "/old"): redirect("/new", status), ("public.com", "/new"): page()})
    assert (await safe_get(client(s), "https://public.com/old"))[0] == "https://public.com/new"


@pytest.mark.parametrize("target", ["http://127.0.0.1/admin", "http://internal.com/", "http://metadata.com/latest",
                                    "http://localhost:8080/", "file:///etc/passwd", "http://[::1]/"])
async def test_redirect_to_private_target_refused_before_connecting(target):
    s = Server({("public.com", "/go"): redirect(target)})
    with pytest.raises(BlockedURL):
        await safe_get(client(s), "https://public.com/go")
    assert s.hosts == ["public.com"]                              # the private hop was never requested


async def test_too_many_redirects():
    s = Server({("public.com", f"/{n}"): redirect(f"/{n + 1}") for n in range(20)})
    with pytest.raises(BlockedURL, match="too many redirects"):
        await safe_get(client(s), "https://public.com/0")
    assert len(s.requests) == sf.MAX_REDIRECTS + 1


async def test_redirect_loop_stops():
    s = Server({("public.com", "/a"): redirect("/b"), ("public.com", "/b"): redirect("/a")})
    with pytest.raises(BlockedURL, match="too many redirects"):
        await safe_get(client(s), "https://public.com/a")


async def test_redirect_body_never_read():
    s = Server({("public.com", "/old"): (302, {"Location": "/new"}, [b"x" * 1000]), ("public.com", "/new"): page()})
    await safe_get(client(s), "https://public.com/old")
    assert s.bodies[0].read == 0 and s.bodies[0].closed


# --- pinned_get: status, type, size --------------------------------------------------

@pytest.mark.parametrize("status", [400, 403, 404, 410, 429, 500, 503])
async def test_non_success_status(status):
    s = Server({("public.com", "/"): (status, HTML, [b"error page"])})
    with pytest.raises(UnsuccessfulFetch, match=str(status)):
        await safe_get(client(s), "https://public.com/")
    assert s.bodies[0].read == 0 and s.bodies[0].closed


@pytest.mark.parametrize("ctype, expected", [
    ("text/html", "text/html"),
    ("text/html; charset=ISO-8859-1", "text/html"),
    ("TEXT/HTML", "text/html"),
    ("text/plain", "text/plain"),
    ("application/xhtml+xml", "application/xhtml+xml"),
])
async def test_allowed_content_types(ctype, expected):
    s = Server({("public.com", "/"): page(headers={"Content-Type": ctype})})
    assert (await safe_get(client(s), "https://public.com/"))[2] == expected


@pytest.mark.parametrize("headers", [{"Content-Type": "application/pdf"}, {"Content-Type": "image/png"},
                                     {"Content-Type": "application/json"}, {"Content-Type": "application/octet-stream"},
                                     {}])
async def test_other_content_types_refused_unread(headers):
    s = Server({("public.com", "/"): (200, headers, [b"%PDF-1.7 ..."])})
    with pytest.raises(InvalidContentType):
        await safe_get(client(s), "https://public.com/")
    assert s.bodies[0].read == 0 and s.bodies[0].closed


async def test_big_content_length_refused_unread():
    s = Server({("public.com", "/"): (200, {**HTML, "Content-Length": str(sf.MAX_BYTES + 1)}, [b"x"])})
    with pytest.raises(BodyTooLarge):
        await safe_get(client(s), "https://public.com/")
    assert s.bodies[0].read == 0 and s.bodies[0].closed


async def test_streaming_cap_without_content_length():
    cap = 1000
    s = Server({("public.com", "/"): (200, HTML, [b"x" * 300] * 100)})     # 30 KB, no Content-Length
    with pytest.raises(BodyTooLarge):
        await safe_get(client(s), "https://public.com/", max_bytes=cap)
    assert s.bodies[0].read <= 1200 and s.bodies[0].closed               # stopped at the 4th chunk


async def test_lying_content_length_still_capped():
    cap = 1000
    s = Server({("public.com", "/"): (200, {**HTML, "Content-Length": "10"}, [b"x" * 600] * 10)})
    with pytest.raises(BodyTooLarge):
        await safe_get(client(s), "https://public.com/", max_bytes=cap)


async def test_exactly_the_cap_is_allowed():
    cap = 1000
    s = Server({("public.com", "/"): (200, HTML, [b"x" * 500, b"x" * 500])})
    assert len((await safe_get(client(s), "https://public.com/", max_bytes=cap))[1]) == 1000


async def test_gzip_bomb_capped_on_decoded_size():
    cap = 10_000
    bomb = gzip.compress(b"\0" * 1_000_000)                              # ~1 KB on the wire, 1 MB decoded
    s = Server({("public.com", "/"): (200, {**HTML, "Content-Encoding": "gzip", "Content-Length": str(len(bomb))},
                                      [bomb])})
    with pytest.raises(BodyTooLarge):
        await safe_get(client(s), "https://public.com/", max_bytes=cap)


async def test_success_closes_the_stream():
    s = Server({("public.com", "/"): page()})
    await safe_get(client(s), "https://public.com/")
    assert s.bodies[0].closed


# --- make_client ---------------------------------------------------------------------

async def test_make_client_settings():
    c = make_client(read_timeout=5, connect_timeout=3)
    try:
        assert c.follow_redirects is False and c.trust_env is False
        assert c.timeout.connect == 3 and c.timeout.read == 5
        assert "Mozilla" in c.headers["User-Agent"]
    finally:
        await c.aclose()



# --- image mode (vision descriptions) ---------------------------------------------------

JPEG = {"Content-Type": "image/jpeg"}


async def test_image_types_only_in_image_mode():
    s = Server({("public.com", "/i.jpg"): (200, JPEG, [b"\xff\xd8jpeg"])})
    with pytest.raises(sf.InvalidContentType):
        await safe_get(client(s), "https://public.com/i.jpg")                 # page mode refuses images
    url, body, ctype = await safe_get(client(s), "https://public.com/i.jpg", sf.IMAGE_CONTENT_TYPES, sf.IMAGE_MAX_BYTES)
    assert body == b"\xff\xd8jpeg" and ctype == "image/jpeg"


async def test_image_mode_refuses_html():
    s = Server({("public.com", "/"): page()})
    with pytest.raises(sf.InvalidContentType):
        await safe_get(client(s), "https://public.com/", sf.IMAGE_CONTENT_TYPES, sf.IMAGE_MAX_BYTES)


async def test_image_cap_with_and_without_content_length():
    big = sf.IMAGE_MAX_BYTES + 1
    s = Server({("public.com", "/a"): (200, {**JPEG, "Content-Length": str(big)}, [b"x"]),
                ("public.com", "/b"): (200, JPEG, [b"x" * 100_000] * 11)})          # 1.1 MB, no Content-Length
    for path in ("/a", "/b"):
        with pytest.raises(BodyTooLarge):
            await safe_get(client(s), f"https://public.com{path}", sf.IMAGE_CONTENT_TYPES, sf.IMAGE_MAX_BYTES)


async def test_image_mode_still_blocks_private_addresses():
    with pytest.raises(sf.BlockedURL):
        await safe_get(client(Server({})), "http://127.0.0.1/i.jpg", sf.IMAGE_CONTENT_TYPES, sf.IMAGE_MAX_BYTES)

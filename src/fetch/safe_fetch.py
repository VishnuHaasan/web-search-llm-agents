import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx

MAX_REDIRECTS = 5
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_PORTS = {80, 443}
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/html",
    "application/xhtml+xml",
}
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif"
}
IMAGE_MAX_BYTES = 1024 * 1024

class FetchError(Exception):
    pass

class UnsuccessfulFetch(FetchError):
    pass

class InvalidContentType(FetchError):
    pass

class BodyTooLarge(FetchError):
    pass

class BlockedURL(FetchError):
    pass


async def resolve_public(host: str, port: int) -> str:
    """Resolve host and return one IP, refusing if ANY resolved address is non-public."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise BlockedURL(f"cannot resolve {host}: {e}")
    addrs = {info[4][0] for info in infos}
    for a in addrs:
        if not ipaddress.ip_address(a.split("%")[0]).is_global:
            raise BlockedURL(f"{host} resolves to non-public {a}")
    return sorted(addrs)[0]


def check_url(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise BlockedURL(f"scheme {parts.scheme!r} not allowed")
    if not parts.hostname:
        raise BlockedURL("no host")
    if parts.username or parts.password:
        raise BlockedURL("credentials in URL")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise BlockedURL(f"port {port} not allowed")
    return parts.hostname, port


async def pinned_get(client: httpx.AsyncClient, url: str, ip: str, allowed_content_types: set[str], max_bytes: int) -> tuple[bytes, httpx.Response, str]:
    """Connect to the already-checked IP, while keeping Host header and TLS name = original host."""
    parts = urlsplit(url)
    ip_host = f"[{ip}]" if ":" in ip else ip
    netloc = ip_host if parts.port is None else f"{ip_host}:{parts.port}"
    req = client.build_request(
        "GET",
        parts._replace(netloc=netloc).geturl(),
        headers={"Host": parts.netloc},
        extensions={"sni_hostname": parts.hostname},
    )
    resp = await client.send(req, stream=True)

    try:
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if resp.is_redirect:
            return bytes(), resp, content_type

        if not resp.is_success:
            raise UnsuccessfulFetch(f"response code {resp.status_code}")

        if content_type not in allowed_content_types:
            raise InvalidContentType(f"invalid content type : {content_type}")

        content_len = resp.headers.get("Content-Length")
        if content_len and content_len.isdigit() and int(content_len) > max_bytes:
            raise BodyTooLarge(f"response too large : {content_len}")

        body = bytearray()
        async for chunk in resp.aiter_bytes():
            body += chunk
            if len(body) > max_bytes:
                raise BodyTooLarge("response too large")
        resp_bytes = bytes(body)
    finally:
        await resp.aclose()
    return resp_bytes, resp, content_type


async def safe_get(client: httpx.AsyncClient, url: str, allowed_types: set[str] =ALLOWED_CONTENT_TYPES, max_bytes=MAX_BYTES) -> tuple[str, bytes, str]:
    """Fetch url, checking every redirect hop. Returns (final_url, response, content_type)."""
    for _ in range(MAX_REDIRECTS + 1):
        host, port = check_url(url)
        ip = await resolve_public(host, port)
        resp_bytes, resp, content_type = await pinned_get(client, url, ip, allowed_types, max_bytes)
        if resp.is_redirect and "location" in resp.headers:
            url = urljoin(url, resp.headers["location"])
            continue
        return url, resp_bytes, content_type
    raise BlockedURL("too many redirects")


def make_client(read_timeout : float, connect_timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=False,
        trust_env=False,
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        headers={"User-Agent": "Mozilla/5.0 (compatible; local-search/0.1)"},
    )

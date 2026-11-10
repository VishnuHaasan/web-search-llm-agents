import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from lxml import html as lhtml

from fetch.safe_fetch import safe_get, FetchError

@dataclass
class Extracted:
    url: str
    final_url: str | None = None
    text: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    error: str | None = None
    markdown: str | None = None
    favicon: str | None = None


def cap(s: str | None, limit: int) -> str | None:
    """Cut s to at most limit chars, at the last line break before the limit when there is one."""
    if s is None or len(s) <= limit:
        return s
    cut = s.rfind("\n", 0, limit)
    return s[:cut if cut > 0 else limit]


def default_favicon(url: str) -> str | None:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}/favicon.ico" if p.scheme in ("http", "https") and p.netloc else None


def find_favicon(body: bytes, base: str) -> str | None:
    """First http(s) <link rel="...icon..."> href, joined with the page URL; else /favicon.ico."""
    try:
        hrefs = lhtml.fromstring(body).xpath("//link[contains(@rel, 'icon')]/@href")
    except Exception:
        hrefs = []
    for h in hrefs:
        u = urljoin(base, h.strip())
        if urlsplit(u).scheme in ("http", "https"):
            return u
    return default_favicon(base)


def _parse_html(body: bytes, url: str, markdown: bool):
    """Runs in a thread: plain extraction, optional markdown extraction, favicon."""
    doc = trafilatura.bare_extraction(
        filecontent=body,
        url=url,
        with_metadata=True,
        include_tables=True,
        date_extraction_params={
            "original_date": True,
            "extensive_search": False,
        }
    )
    md = None
    if markdown and doc is not None and doc.text:
        d = trafilatura.bare_extraction(filecontent=body, url=url, include_tables=True,
                                        include_formatting=True, include_links=True)
        md = d.text if d is not None and d.text else None
    return doc, md, find_favicon(body, url)

class Extractor:
    def __init__(self, client: httpx.AsyncClient, page_timeout: float, min_text_filter: int, concurrency: int,
                 max_page_chars: int = 100_000) -> None:
        self.page_timeout = page_timeout
        self.max_page_chars = max_page_chars
        self.client = client
        self.min_text_filter = min_text_filter
        self.concurrency = concurrency

    async def extract_many(self, urls: list[str], remaining: float | None = None,
                           markdown: bool = False) -> list[Extracted]:
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [asyncio.create_task(self.extract_one(url, sem, markdown)) for url in urls]
        if not tasks:
            return []
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return [t.result() if t in done
                else Extracted(url=url, error="request budget time exceeded", favicon=default_favicon(url))
                for t, url in zip(tasks, urls)]

    async def extract_one(self, url: str, sem: asyncio.Semaphore, markdown: bool = False) -> Extracted:
        out = await self._extract_one(url, sem, markdown)
        if out.favicon is None:
            out.favicon = default_favicon(out.final_url or url)
        return out

    async def _extract_one(self, url: str, sem: asyncio.Semaphore, markdown: bool) -> Extracted:

        async with sem:
            try:
                async with asyncio.timeout(self.page_timeout):
                    final_url, body, content_type = await safe_get(self.client, url)
                if content_type == "text/plain":
                    text = body.decode("utf-8", errors="replace")
                    return Extracted(
                        url=url,
                        final_url=final_url,
                        text=cap(text, self.max_page_chars),
                    )
                doc, md, favicon = await asyncio.to_thread(_parse_html, body, final_url, markdown)
                if doc is None or not doc.text or len(doc.text.strip()) < self.min_text_filter:
                    return Extracted(
                        url=url,
                        final_url=final_url,
                        error="no main text",
                        favicon=favicon,
                    )

                published_at = datetime.strptime(doc.date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc) if doc.date else None
                title = doc.title
                return Extracted(
                    url=url,
                    final_url=final_url,
                    text=cap(doc.text, self.max_page_chars),
                    title=title,
                    published_at=published_at,
                    markdown=cap(md, self.max_page_chars),
                    favicon=favicon,
                )
            except TimeoutError:
                return Extracted(
                    url=url,
                    error="Timeout error",
                )
            except FetchError as e:
                return Extracted(
                    url=url,
                    error=str(e),
                )
            except httpx.HTTPError as e:
                return Extracted(
                    url=url,
                    error=f"{type(e).__name__}: {e}",
                )
            except Exception as e:
                return Extracted(
                    url=url,
                    error=f"Exception fetching page : {type(e).__name__}: {e}",
                )



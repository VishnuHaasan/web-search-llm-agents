import json
from pathlib import Path

import httpx
import pytest

from search.filters import Filters
from search.fourget.provider import FourGetProvider
from search.schema import Candidate, ProviderHit, FourGetResponse

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_page(name: str, scraper: str = "ddg") -> FourGetResponse:
    page = FourGetResponse.model_validate(load_fixture(name))
    page.scraper = scraper
    return page


def make_provider(handler) -> FourGetProvider:
    # every request goes to handler(request) instead of the network
    client = httpx.AsyncClient(base_url="http://4get/api/v1", transport=httpx.MockTransport(handler))
    return FourGetProvider(client)


def cand(url: str, rank: int, canon: str | None = None, engine: str = "ddg", **kw) -> Candidate:
    return Candidate(
        url=url,
        canonical_url=canon or Filters.canonicalize_url(url),
        title=kw.pop("title", f"{engine}#{rank}"),
        hits=[ProviderHit(provider="4get", engine=engine, rank=rank)],
        **kw,
    )


@pytest.fixture(scope="session", autouse=True)
def filters_dir(tmp_path_factory) -> Path:
    # a tiny data dir: the real data/blocklists is gitignored and empty on a fresh clone
    d = tmp_path_factory.mktemp("data")
    (d / "blocklists").mkdir()
    (d / "blocklists" / "test.txt").write_text(
        "*://*.code-examples.net/*\n*://*.spam-site.xyz/*\n*://*.allowed-spam.net/*\n", encoding="utf-8")
    (d / "no-extract.txt").write_text("# test list\nyoutube.com\nreddit.com\n", encoding="utf-8")
    (d / "allow-list.txt").write_text("allowed-spam.net\n", encoding="utf-8")
    Filters.load(d)
    return d

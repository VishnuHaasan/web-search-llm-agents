from datetime import date, timezone

import httpx
import pytest

from conftest import load_fixture, load_page, make_provider
from models.schema import SearchQuery
from search.filters import Filters
from search.schema import Spelling
from utils.errors import FourGetError

Q = SearchQuery(query="asyncio")


@pytest.fixture
def provider():
    return make_provider(lambda r: httpx.Response(500))


class Fake4get:
    """A MockTransport handler that records requests and answers per scraper/endpoint."""

    def __init__(self, by_scraper=None, default="web_ddg.json", images="images_ddg.json"):
        self.by_scraper, self.default, self.images = by_scraper or {}, default, images
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/images"):
            return self._answer(self.images, request)
        return self._answer(self.by_scraper.get(request.url.params.get("scraper"), self.default), request)

    @staticmethod
    def _answer(what, request):
        if isinstance(what, Exception):
            raise what
        if isinstance(what, int):
            return httpx.Response(what)
        return httpx.Response(200, json=load_fixture(what))

    @property
    def scrapers(self):
        return [r.url.params.get("scraper") for r in self.requests if not r.url.path.endswith("/images")]


# --- mapping from saved responses -------------------------------------------------

def test_web_candidates(provider):
    page = load_page("web_ddg.json")
    raw = [r.url for r in page.web]
    cands = provider.to_candidates(Q, page, 0)
    assert cands
    for c in cands:
        hit = c.hits[0]
        assert hit.rank == raw.index(c.url) + 1          # numbered before skipping
        assert hit.provider == "4get" and hit.engine == "ddg"
        assert c.kind == "web" and Filters.is_fetchable(c.url)
        assert c.published_at is None or c.published_at.tzinfo == timezone.utc


def test_rank_offset(provider):
    assert provider.to_candidates(Q, load_page("web_ddg.json"), 10)[0].hits[0].rank > 10


def test_news_candidates(provider):
    cands = provider.to_candidates(Q, load_page("news_ddg.json"), 0)
    assert cands and {c.kind for c in cands} == {"news"}
    assert all(c.published_at for c in cands)
    assert any(c.thumbnail for c in cands)


def test_engine_is_the_scraper_used(provider):
    assert provider.to_candidates(Q, load_page("web_ddg.json", "startpage"), 0)[0].hits[0].engine == "startpage"


def test_instant_answer(provider):
    answers = provider.to_instant_answers(load_page("web_answer_ddg.json"))
    assert len(answers) == 1
    assert answers[0].title == "Python (programming language)"
    assert answers[0].text.startswith("Python is a high-level")


def test_spelling_uses_corrected_query(provider):
    assert provider.spelling_of(load_page("web_spelling_ddg.json")) == "python asyncio"


@pytest.mark.parametrize("using, expected", [
    ("python asyncio site:docs.python.org -site:medium.com", "python asyncio"),
    ("python asyncio (site:a.com OR site:b.com)", "python asyncio"),
    ("site:a.com", None),
    ("", None),
])
def test_spelling_strips_operators(provider, using, expected):
    page = load_page("web_ddg.json")
    page.spelling = Spelling(type="including", using=using)
    assert provider.spelling_of(page) == expected


def test_no_correction(provider):
    page = load_page("web_ddg.json")
    page.spelling = Spelling(type="no_correction")
    assert provider.spelling_of(page) is None


@pytest.mark.parametrize("fixture", ["web_images_ddg.json", "images_ddg.json"])
def test_images_largest_source_and_cap(provider, fixture):
    page = load_page(fixture)
    images = provider.to_images(page)
    assert len(images) == 10
    first = page.image[0]
    largest = max(first.source, key=lambda s: (s.width or 0) * (s.height or 0))
    assert images[0].url == largest.url and images[0].source_page == first.url


@pytest.mark.parametrize("fixture", ["web_images_ddg.json", "images_ddg.json"])
def test_images_keep_smallest_source_as_thumbnail(provider, fixture):
    page = load_page(fixture)
    first = page.image[0]
    smallest = min(first.source, key=lambda s: (s.width or 0) * (s.height or 0))
    img = provider.to_images(page)[0]
    assert img.thumbnail == smallest.url
    assert len(first.source) == 1 or img.thumbnail != img.url


def test_image_files_are_not_dropped(provider):
    # is_fetchable rejects .png/.svg; images must not go through it
    assert any(i.url.endswith(".png") for i in provider.to_images(load_page("web_images_ddg.json")))


# --- fetch_page against a fake 4get -------------------------------------------------

async def test_one_request_when_healthy():
    fake = Fake4get()
    page, cursor = await make_provider(fake).fetch_page(Q, "asyncio", None)
    assert fake.scrapers == ["ddg"]
    assert page.scraper == "ddg" and cursor == f"ddg|{page.npt}"
    assert fake.requests[0].url.path == "/api/v1/web"


async def test_request_params():
    fake = Fake4get()
    q = SearchQuery(query="asyncio", safe_search=True, country="us", start_date=date(2025, 1, 1), end_date=date(2025, 1, 31))
    await make_provider(fake).fetch_page(q, "asyncio site:x.com", None)
    params = dict(fake.requests[0].url.params)
    assert params == {"s": "asyncio site:x.com", "scraper": "ddg", "country": "us", "nsfw": "no",
                      "newer": "2025-01-01", "older": "2025-01-31"}


async def test_no_empty_params():
    fake = Fake4get()
    await make_provider(fake).fetch_page(Q, "asyncio", None)
    assert "nsfw" not in fake.requests[0].url.params and "country" not in fake.requests[0].url.params


async def test_news_endpoint():
    fake = Fake4get(default="news_ddg.json")
    await make_provider(fake).fetch_page(SearchQuery(query="python", topic="news"), "python", None)
    assert fake.requests[0].url.path == "/api/v1/news"


async def test_page_two_sends_only_the_token():
    fake = Fake4get()
    p = make_provider(fake)
    page, _ = await p.fetch_page(Q, "asyncio", "startpage|tok123")
    assert dict(fake.requests[0].url.params) == {"npt": "tok123"}
    assert page.scraper == "startpage"


async def test_blocked_scraper_falls_back():
    fake = Fake4get({"ddg": "error_captcha.json"})
    p = make_provider(fake)
    page, cursor = await p.fetch_page(Q, "asyncio", None)
    assert fake.scrapers == ["ddg", "startpage"]
    assert page.scraper == "startpage" and cursor.startswith("startpage|")
    assert p.scraper_pool.scraper_states["ddg"].fail_count == 1


async def test_cooled_scraper_skipped_on_next_request():
    fake = Fake4get({"ddg": "error_captcha.json"})
    p = make_provider(fake)
    await p.fetch_page(Q, "asyncio", None)
    fake.requests.clear()
    await p.fetch_page(Q, "asyncio", None)
    assert fake.scrapers == ["startpage"]


async def test_gives_up_after_two_scrapers():
    fake = Fake4get({"ddg": "error_captcha.json", "startpage": "web_brave.json"})
    with pytest.raises(FourGetError) as e:
        await make_provider(fake).fetch_page(Q, "asyncio", None)
    assert fake.scrapers == ["ddg", "startpage"] and e.value.kind == FourGetError.BLOCKED


async def test_4get_down_cools_nothing():
    p = make_provider(Fake4get(default=httpx.ConnectError("refused")))
    with pytest.raises(FourGetError) as e:
        await p.fetch_page(Q, "asyncio", None)
    assert e.value.kind == FourGetError.DOWN
    assert all(s.fail_count == 0 for s in p.scraper_pool.scraper_states.values())


@pytest.mark.parametrize("answer, kind", [
    (httpx.ReadTimeout("slow"), FourGetError.TRANSIENT),
    (httpx.ReadError("dropped"), FourGetError.TRANSIENT),
    (503, FourGetError.TRANSIENT),
    (429, FourGetError.BLOCKED),
])
async def test_error_kinds(answer, kind):
    p = make_provider(Fake4get(default=answer))
    with pytest.raises(FourGetError) as e:
        await p.fetch_page(Q, "asyncio", None)
    assert e.value.kind == kind
    assert p.scraper_pool.scraper_states["ddg"].fail_count == 1


async def test_bad_token_is_invalid_and_not_retried():
    fake = Fake4get(default="error_bad_npt.json")
    p = make_provider(fake)
    with pytest.raises(FourGetError) as e:
        await p.fetch_page(Q, "asyncio", "ddg|garbage")
    assert e.value.kind == FourGetError.INVALID and len(fake.requests) == 1
    assert p.scraper_pool.scraper_states["ddg"].fail_count == 0


async def test_page_two_failure_is_not_retried():
    fake = Fake4get(default="error_captcha.json")
    p = make_provider(fake)
    with pytest.raises(FourGetError):
        await p.fetch_page(Q, "asyncio", "ddg|tok")
    assert len(fake.requests) == 1 and p.scraper_pool.scraper_states["ddg"].fail_count == 1


async def test_images_requested_only_when_asked():
    fake = Fake4get()
    p = make_provider(fake)
    await p.fetch_page(Q, "asyncio", None)
    assert not any(r.url.path.endswith("/images") for r in fake.requests)
    page, _ = await p.fetch_page(SearchQuery(query="asyncio", include_images=True), "asyncio", None)
    assert any(r.url.path.endswith("/images") for r in fake.requests)
    assert len(page.image) == 15


@pytest.mark.parametrize("images", [500, httpx.ConnectError("refused")])
async def test_failed_images_keep_web_results(images):
    page, _ = await make_provider(Fake4get(images=images)).fetch_page(
        SearchQuery(query="asyncio", include_images=True), "asyncio", None)
    assert len(page.web) == 10 and page.image == []


async def test_search_end_to_end():
    b = await make_provider(Fake4get()).search(SearchQuery(query="asyncio", max_results=5, include_images=True))
    assert b.candidates and not b.errors
    assert {h.engine for c in b.candidates for h in c.hits} == {"ddg"}
    assert len(b.images) == 10

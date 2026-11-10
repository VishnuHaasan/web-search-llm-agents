import asyncio

from conftest import cand
from models.schema import SearchQuery
from search.query import SearchProvider
from search.schema import Candidate, ImageCandidate, InstantAnswer


class FakeProvider(SearchProvider[dict]):
    """Pages are plain dicts; the cursor is the next page number as a string."""
    name = "fake"

    def __init__(self, pages: dict[str, list[list[str]]], fail_on: set[tuple[str, int]] = frozenset(),
                 delay: float = 0.0, max_pages: int = 3):
        super().__init__(max_pages)
        self.pages, self.fail_on, self.delay = pages, fail_on, delay
        self.calls: list[tuple[str, int]] = []

    def build_queries(self, q: SearchQuery) -> list[str]:
        return list(self.pages)

    async def fetch_page(self, q, query, cursor):
        n = int(cursor or 0)
        self.calls.append((query, n))
        if self.delay:
            await asyncio.sleep(self.delay)
        if (query, n) in self.fail_on:
            raise RuntimeError("captcha")
        urls = self.pages[query][n]
        nxt = str(n + 1) if n + 1 < len(self.pages[query]) else None
        page = {"urls": urls, "images": [f"https://img/{query}/{n}"], "answer": f"{query}-{n}", "spelling": f"fix-{query}"}
        return page, nxt

    def to_candidates(self, q, page, rank_offset) -> list[Candidate]:
        return [cand(u, rank_offset + i + 1, engine="fake") for i, u in enumerate(page["urls"])]

    def to_images(self, page):
        return [ImageCandidate(url=u, provider=self.name) for u in page["images"]]

    def to_instant_answers(self, page):
        return [InstantAnswer(provider=self.name, title="t", text=page["answer"])]

    def spelling_of(self, page):
        return page["spelling"]


def urls(prefix: str, n: int, start: int = 0) -> list[str]:
    return [f"https://{prefix}{i}.com/" for i in range(start, start + n)]


async def test_stops_when_share_reached():
    # max_results=2, basic -> target 4; the first page already has 4
    p = FakeProvider({"q": [urls("a", 4), urls("a", 4, 4)]})
    b = await p.search(SearchQuery(query="q", max_results=2))
    assert p.calls == [("q", 0)] and len(b.candidates) == 4


async def test_stops_without_cursor():
    p = FakeProvider({"q": [urls("a", 2)]})
    await p.search(SearchQuery(query="q", max_results=10))
    assert p.calls == [("q", 0)]


async def test_stops_when_page_has_nothing_new():
    p = FakeProvider({"q": [urls("a", 2), urls("a", 2), urls("a", 2, 2)]})
    b = await p.search(SearchQuery(query="q", max_results=10))
    assert p.calls == [("q", 0), ("q", 1)] and len(b.candidates) == 2


async def test_stops_at_max_pages():
    p = FakeProvider({"q": [urls("a", 2, i * 2) for i in range(10)]})
    b = await p.search(SearchQuery(query="q", max_results=10))
    assert len(p.calls) == 3 and len(b.candidates) == 6


async def test_stops_at_time_budget(monkeypatch):
    monkeypatch.setattr(FakeProvider, "_TIME_BUDGET", 0.15)
    p = FakeProvider({"q1": [urls("a", 1, i) for i in range(3)], "q2": [urls("b", 1, i) for i in range(3)]}, delay=0.1)
    await p.search(SearchQuery(query="q", max_results=10))
    assert len(p.calls) == 2          # the check runs before each fetch, so q2 doesn't start a third


async def test_ranks_count_duplicates_and_continue_across_pages():
    p = FakeProvider({"q": [["https://a.com/", "https://b.com/"], ["https://b.com/", "https://c.com/"]]})
    b = await p.search(SearchQuery(query="q", max_results=10))
    assert {c.canonical_url: c.hits[0].rank for c in b.candidates} == {"a.com": 1, "b.com": 2, "c.com": 4}


async def test_seen_is_shared_and_ranks_restart_per_query():
    p = FakeProvider({"q1": [["https://a.com/", "https://b.com/"]], "q2": [["https://b.com/", "https://c.com/"]]})
    b = await p.search(SearchQuery(query="q", max_results=10))
    assert [(c.canonical_url, c.hits[0].rank) for c in b.candidates] == [("a.com", 1), ("b.com", 2), ("c.com", 2)]
    assert b.effective_query == ["q1", "q2"]


async def test_share_is_split_between_queries():
    # target 4, two queries -> 2 each
    p = FakeProvider({"q1": [urls("a", 2), urls("a", 2, 2)], "q2": [urls("b", 2), urls("b", 2, 2)]})
    await p.search(SearchQuery(query="q", max_results=2))
    assert p.calls == [("q1", 0), ("q2", 0)]


async def test_error_on_page_two_keeps_page_one():
    p = FakeProvider({"q": [urls("a", 2), urls("a", 2, 2)], "q2": [urls("b", 2)]}, fail_on={("q", 1)})
    b = await p.search(SearchQuery(query="q", max_results=10))
    assert len(b.candidates) == 4                       # 2 from q page 1, 2 from q2
    assert len(b.errors) == 1 and "'q'" in b.errors[0].message and b.errors[0].provider == "fake"


async def test_extras_from_first_pages_only():
    p = FakeProvider({"q1": [urls("a", 1), urls("a", 1, 1)], "q2": [urls("b", 1), urls("b", 1, 1)]})
    b = await p.search(SearchQuery(query="q", max_results=10, include_images=True, include_answer=True))
    assert [i.url for i in b.images] == ["https://img/q1/0", "https://img/q2/0"]    # page 0 of every query
    assert [a.text for a in b.instant_answers] == ["q1-0"]                          # first query only
    assert b.spelling_correction == "fix-q1"


async def test_extras_off_by_default():
    b = await FakeProvider({"q": [urls("a", 1)]}).search(SearchQuery(query="q"))
    assert b.images == [] and b.instant_answers == []


async def test_no_queries():
    b = await FakeProvider({}).search(SearchQuery(query="q"))
    assert b.candidates == [] and b.effective_query == []


async def test_elapsed_ms_is_milliseconds():
    b = await FakeProvider({"q": [urls("a", 1)]}, delay=0.05).search(SearchQuery(query="q"))
    assert 40 < b.elapsed_ms < 2000

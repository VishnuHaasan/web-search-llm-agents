from datetime import datetime, timezone

import pytest

from conftest import cand
from models.schema import SearchQuery
from search.merge import merge_batches
from search.schema import SearchBatch, ImageCandidate, InstantAnswer, ProviderError


def batch(candidates=(), **kw) -> SearchBatch:
    return SearchBatch(query="q", effective_query=kw.pop("effective_query", ["q"]), candidates=list(candidates), **kw)


def test_same_page_from_two_batches_is_merged():
    a = batch([cand("http://www.docs.python.org/a/", 1, engine="ddg")])
    b = batch([cand("https://docs.python.org/a", 3, engine="searx")])
    m = merge_batches([a, b], SearchQuery(query="q"))
    assert len(m.candidates) == 1
    assert {h.engine for h in m.candidates[0].hits} == {"ddg", "searx"}


def test_rrf_score():
    a = batch([cand("https://a.com/1", 1), cand("https://b.com/1", 2)])
    b = batch([cand("https://a.com/1", 3, engine="searx")])
    m = merge_batches([a, b], SearchQuery(query="q"))
    scores = {c.canonical_url: c.prior_score for c in m.candidates}
    assert scores["a.com/1"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b.com/1"] == pytest.approx(1 / 62)


def test_sorted_best_first():
    m = merge_batches([batch([cand(f"https://site{r}.com/", r) for r in (5, 1, 3)])], SearchQuery(query="q"))
    assert [c.hits[0].rank for c in m.candidates] == [1, 3, 5]


def test_found_twice_beats_single_first_place():
    a = batch([cand("https://one.com/", 1), cand("https://both.com/", 4)])
    b = batch([cand("https://both.com/", 4, engine="searx")])
    m = merge_batches([a, b], SearchQuery(query="q"))
    assert m.candidates[0].canonical_url == "both.com"


def test_tie_broken_by_best_rank():
    # x and y both have ranks 1 and 2, so equal scores and equal best rank
    a = batch([cand("https://x.com/", 2), cand("https://y.com/", 1)])
    b = batch([cand("https://x.com/", 1, engine="searx"), cand("https://y.com/", 2, engine="searx")])
    m = merge_batches([a, b], SearchQuery(query="q"))
    assert m.candidates[0].prior_score == m.candidates[1].prior_score
    assert m.candidates[0].canonical_url == "x.com"   # first seen wins when best ranks also tie


def test_merged_fields():
    early, late = datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2024, 5, 1, tzinfo=timezone.utc)
    a = batch([cand("http://example.com/z", 4, title="worse", snippet="short", published_at=late)])
    b = batch([cand("https://example.com/z", 1, engine="searx", title="best", snippet="a much longer snippet",
                    published_at=early, kind="news", thumbnail="https://img/t.png")])
    c = merge_batches([a, b], SearchQuery(query="q")).candidates[0]
    assert c.url == "https://example.com/z"
    assert c.title == "best"
    assert c.snippet == "a much longer snippet"
    assert c.published_at == early
    assert c.kind == "news"
    assert c.thumbnail == "https://img/t.png"


def test_https_check_is_a_prefix_not_a_substring():
    a = batch([cand("http://example.com/https-guide", 1, canon="example.com/z")])
    b = batch([cand("https://example.com/z", 2, canon="example.com/z")])
    assert merge_batches([a, b], SearchQuery(query="q")).candidates[0].url == "https://example.com/z"


def test_blocklisted_and_excluded_removed():
    m = merge_batches([batch([cand("https://code-examples.net/a", 1), cand("https://docs.python.org/a", 2),
                              cand("https://realpython.com/a", 3)])],
                      SearchQuery(query="q", exclude_domains=["python.org"]))
    assert [c.canonical_url for c in m.candidates] == ["realpython.com/a"]


def test_domain_cap_moves_extras_down():
    cands = [cand(f"https://www.geeksforgeeks.org/{n}", n + 1) for n in range(5)]
    cands += [cand("https://realpython.com/lc", 6), cand("https://docs.python.org/t", 7)]
    m = merge_batches([batch(cands)], SearchQuery(query="q", max_results=10))
    domains = [c.canonical_url.split("/")[0] for c in m.candidates]
    assert domains == ["geeksforgeeks.org"] * 3 + ["realpython.com", "docs.python.org"] + ["geeksforgeeks.org"] * 2


def test_domain_cap_keeps_order_for_a_single_site():
    cands = [cand(f"https://docs.python.org/{n}", n + 1) for n in range(8)]
    m = merge_batches([batch(cands)], SearchQuery(query="q", max_results=4, include_domains=["docs.python.org"]))
    assert [c.hits[0].rank for c in m.candidates] == list(range(1, 9))


def test_trimmed_to_candidate_target():
    # basic depth, max_results=3 -> 2 * 3 = 6
    m = merge_batches([batch([cand(f"https://site{n}.com/", n + 1) for n in range(20)])],
                      SearchQuery(query="q", max_results=3))
    assert len(m.candidates) == 6


def test_batch_fields():
    img = lambda n: ImageCandidate(url=f"https://i/{n}", provider="4get")
    a = batch(effective_query=["q", "q site:x"], elapsed_ms=900, images=[img(n) for n in range(8)],
              errors=[ProviderError(provider="4get", message="captcha")])
    b = batch(effective_query=["q"], elapsed_ms=1200, images=[img(n) for n in range(5, 12)],
              spelling_correction="python asyncio",
              instant_answers=[InstantAnswer(provider="4get", title="t", text="answer")])
    m = merge_batches([a, b], SearchQuery(query="q"))
    assert m.query == "q"
    assert m.effective_query == ["q", "q site:x"]
    assert m.elapsed_ms == 1200                     # max, providers run in parallel
    assert m.spelling_correction == "python asyncio"
    assert [i.url for i in m.images] == [f"https://i/{n}" for n in range(10)]
    assert len(m.errors) == 1 and len(m.instant_answers) == 1


def test_no_batches():
    m = merge_batches([], SearchQuery(query="q"))
    assert m.candidates == [] and m.elapsed_ms == 0

import math

import pytest

from conftest import cand
from models.schema import SearchQuery
from rank.rank import Ranker

Q = SearchQuery(query="asyncio task groups")
ADV = SearchQuery(query="asyncio task groups", search_depth="advanced", chunks_per_source=2)


class FakeBM25:
    """Scores each chunk by looking up its text, so tests control the numbers exactly."""

    def __init__(self, by_text):
        self.by_text = by_text

    def scores(self, query, docs):
        return [self.by_text.get(d, 0.0) for d in docs]


class FakeReranker:
    """Cross-encoder stand-in: logit looked up by chunk text; records every call."""

    def __init__(self, by_text):
        self.by_text = by_text
        self.calls = []

    def scores(self, query, docs):
        self.calls.append((query, list(docs)))
        return [self.by_text.get(d, -11.0) for d in docs]


def ranker(alpha=0.5, size=500, overlap=0, fake=None, ce=None, per_page=5):
    # model_name="" fails to load fast, so the real model is never downloaded in tests
    r = Ranker(k1=1.5, b=0.75, max_chunk_size=size, chunk_overlap=overlap, alpha=alpha,
               model_name="", model_cache_dir="", rerank_per_page=per_page)
    if fake is not None:
        r.bm25 = FakeBM25(fake)
    if ce is not None:
        r.reranker = FakeReranker(ce)
    return r


def sig(x):
    return 1 / (1 + math.exp(-x))


def page(name, rank, text=None, snippet=None):
    return cand(f"https://{name}.com/", rank, prior_score=1 / (60 + rank), text=text, snippet=snippet)


def names(cands):
    return [c.url[8:-5] for c in cands]


# --- 3d / 3e: blending and order --------------------------------------------------

def test_worked_example():
    cs = [page("a", 1, "A"), page("b", 2, "B"), page("c", 3, "C"), page("d", 4, "D")]
    out = ranker(fake={"A": 3.0, "B": 12.0, "C": 0.0, "D": 9.0}).rank_candidates(cs, Q)
    assert names(out) == ["b", "a", "d", "c"]
    assert [round(c.score, 3) for c in out] == [0.828, 0.625, 0.375, 0.161]


def test_alpha_zero_keeps_engine_order():
    cs = [page("a", 1, "A"), page("b", 2, "B"), page("c", 3, "C")]
    out = ranker(alpha=0, fake={"A": 1.0, "B": 9.0, "C": 5.0}).rank_candidates(cs, Q)
    assert names(out) == ["a", "b", "c"]


def test_alpha_one_sorts_by_bm25_only():
    cs = [page("a", 1, "A"), page("b", 2, "B"), page("c", 3, "C")]
    out = ranker(alpha=1, fake={"A": 1.0, "B": 9.0, "C": 5.0}).rank_candidates(cs, Q)
    assert names(out) == ["b", "c", "a"]


def test_ties_broken_by_prior():
    cs = [page("b", 2, "X"), page("a", 1, "X")]
    out = ranker(alpha=1, fake={"X": 4.0}).rank_candidates(cs, Q)
    assert names(out) == ["a", "b"]


def test_no_candidates():
    assert ranker().rank_candidates([], Q) == []


def test_single_candidate_without_match_keeps_a_score():
    # all priors equal -> prior_part 1.0, so a page is never left with score 0 just for being alone
    [c] = ranker().rank_candidates([page("a", 1, "nothing relevant here")], Q)
    assert c.score == pytest.approx(0.5)


def test_no_match_anywhere_falls_back_to_engine_order():
    cs = [page("a", 1, "x"), page("b", 2, "y"), page("c", 3, "z")]
    assert names(ranker().rank_candidates(cs, Q)) == ["a", "b", "c"]


# --- 3c: best chunk and content -------------------------------------------------------

def test_best_chunk_not_sum():
    # a: many weak chunks; b: one strong chunk; same engine prior
    weak = "\n".join(f"asyncio note {n} " + "filler " * 60 for n in range(10))
    strong = "asyncio task groups: task groups run tasks together, and task groups cancel on failure."
    cs = [cand("https://a.com/", 1, prior_score=0.016, text=weak),
          cand("https://b.com/", 1, prior_score=0.016, text=strong)]
    assert names(ranker().rank_candidates(cs, Q)) == ["b", "a"]


def test_basic_content_is_the_single_best_chunk():
    text = "intro\ntaskgroup part\noutro"
    [c] = ranker(size=15, fake={"intro": 1.0, "taskgroup part": 9.0, "outro": 2.0}).rank_candidates(
        [page("a", 1, text)], Q)
    assert c.content == "taskgroup part"


def test_advanced_content_is_top_k_in_page_order():
    text = "first\nsecond\nthird\nfourth"
    [c] = ranker(size=8, fake={"first": 1.0, "second": 2.0, "third": 0.5, "fourth": 9.0}).rank_candidates(
        [page("a", 1, text)], ADV)
    assert c.content == "second [...] fourth"


def test_no_match_uses_snippet():
    [c] = ranker(fake={}).rank_candidates([page("a", 1, "menu\nbody", snippet="engine snippet")], Q)
    assert c.content == "engine snippet"


def test_no_match_no_snippet_uses_first_chunk():
    [c] = ranker(size=5, fake={}).rank_candidates([page("a", 1, "menu\nbody")], Q)
    assert c.content == "menu"


def test_no_text_no_snippet_gives_empty_content():
    [c] = ranker().rank_candidates([page("a", 1)], Q)
    assert c.content == ""


def test_failed_extraction_still_ranks_on_its_snippet():
    # apply_extraction sets text = snippet when extraction failed
    s = "asyncio task groups explained"
    cs = [page("a", 1, "unrelated words only"), page("b", 2, s, snippet=s), page("c", 3, "more unrelated words")]
    out = ranker().rank_candidates(cs, Q)
    assert names(out) == ["b", "a", "c"] and out[0].content == s


# --- schema -----------------------------------------------------------------------------

def test_chunks_per_source_is_the_tavily_name():
    assert SearchQuery(query="q", chunks_per_source=1).chunks_per_source == 1


@pytest.mark.parametrize("n", [0, 4])
def test_chunks_per_source_range(n):
    with pytest.raises(ValueError):
        SearchQuery(query="q", chunks_per_source=n)


# --- step 5: cross-encoder rerank (advanced) ----------------------------------------

# page B from the worked example: 6 chunks, BM25 likes the quiz, the cross-encoder likes "errors"
B_TEXT = "intro\ninstall\ntaskgroup\nquiz\nerrors\noutro"
B_BM25 = {"intro": 0.4, "install": 0.0, "taskgroup": 12.0, "quiz": 5.1, "errors": 3.0, "outro": 1.2}
B_CE = {"taskgroup": 1.9, "quiz": -10.8, "errors": 4.3}


def test_model_load_failure_falls_back_to_none():
    assert ranker().reranker is None


def test_content_picked_by_ce_in_page_order():
    r = ranker(size=10, fake=B_BM25, ce=B_CE, per_page=3)
    [c] = r.rank_candidates([page("b", 1, B_TEXT)], ADV)
    assert c.content == "taskgroup [...] errors"


def test_shortlist_is_top_bm25_chunks_only():
    r = ranker(size=10, fake=B_BM25, ce=B_CE, per_page=3)
    r.rank_candidates([page("b", 1, B_TEXT)], ADV)
    [(_, sent)] = r.reranker.calls
    assert sorted(sent) == ["errors", "quiz", "taskgroup"]


def test_chunk_outside_shortlist_never_picked():
    # "outro" would win on CE, but BM25 ranks it 4th, so with a shortlist of 3 it is never scored
    r = ranker(size=10, fake=B_BM25, ce={**B_CE, "outro": 9.0}, per_page=3)
    [c] = r.rank_candidates([page("b", 1, B_TEXT)], ADV)
    assert "outro" not in c.content


def test_bigger_shortlist_lets_ce_pull_in_a_lower_bm25_chunk():
    r = ranker(size=10, fake=B_BM25, ce={**B_CE, "outro": 9.0}, per_page=5)
    [c] = r.rank_candidates([page("b", 1, B_TEXT)], ADV)
    assert c.content == "errors [...] outro"


def test_shortlist_at_least_chunks_per_source():
    q = SearchQuery(query="asyncio task groups", search_depth="advanced", chunks_per_source=3)
    r = ranker(size=10, fake=B_BM25, ce=B_CE, per_page=2)
    r.rank_candidates([page("b", 1, B_TEXT)], q)
    assert len(r.reranker.calls[0][1]) == 3


def test_text_score_is_sigmoid_not_divided_by_max():
    # every page irrelevant: the top page must not be lifted to 1.0
    cs = [page("a", 1, "x"), page("b", 2, "y")]
    out = ranker(alpha=1, fake={}, ce={"x": -9.0, "y": -10.0}).rank_candidates(cs, ADV)
    assert out[0].score == pytest.approx(sig(-9.0)) and out[0].score < 0.01


def test_blend_uses_sigmoid():
    cs = [page("a", 1, "A"), page("b", 2, "B")]
    out = ranker(fake={}, ce={"A": -9.5, "B": 4.3}).rank_candidates(cs, ADV)
    # a: 0.5*0.00007 + 0.5*1.0 = 0.50004   b: 0.5*0.987 + 0.5*0.0 = 0.4932
    assert names(out) == ["a", "b"]
    assert out[0].score == pytest.approx(0.5 * sig(-9.5) + 0.5 * 1.0)
    assert out[1].score == pytest.approx(0.5 * sig(4.3) + 0.5 * 0.0)


def test_ce_beats_bm25_across_pages():
    # BM25 prefers a, the cross-encoder prefers b
    cs = [page("a", 1, "A"), page("b", 1, "B")]
    out = ranker(alpha=1, fake={"A": 9.0, "B": 1.0}, ce={"A": -5.0, "B": 3.0}).rank_candidates(cs, ADV)
    assert names(out) == ["b", "a"]


def test_negative_ce_uses_snippet():
    [c] = ranker(fake={"body": 5.0}, ce={"body": -3.0}).rank_candidates(
        [page("a", 1, "body", snippet="engine snippet")], ADV)
    assert c.content == "engine snippet"


def test_page_without_text_does_not_crash():
    cs = [page("a", 1, snippet="snip"), page("b", 2, "B")]
    r = ranker(fake={}, ce={"B": 2.0})
    out = r.rank_candidates(cs, ADV)
    a = next(c for c in out if c.url == "https://a.com/")
    assert a.content == "snip" and len(r.reranker.calls) == 1


def test_site_operator_removed_from_ce_query():
    q = SearchQuery(query="asyncio site:docs.python.org", search_depth="advanced")
    r = ranker(fake={}, ce={})
    r.rank_candidates([page("a", 1, "A")], q)
    assert r.reranker.calls[0][0] == "asyncio"


def test_basic_never_calls_reranker():
    r = ranker(fake={"A": 1.0}, ce={"A": 5.0})
    r.rank_candidates([page("a", 1, "A")], Q)
    assert r.reranker.calls == []


def test_no_reranker_advanced_uses_bm25():
    text = "first\nsecond\nthird\nfourth"
    [c] = ranker(size=8, fake={"first": 1.0, "second": 2.0, "third": 0.5, "fourth": 9.0}).rank_candidates(
        [page("a", 1, text)], ADV)
    assert c.content == "second [...] fourth" and c.score == pytest.approx(1.0)

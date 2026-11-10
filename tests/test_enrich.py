from datetime import datetime, timedelta, timezone

from conftest import cand
from fetch.extract import Extracted
from models.schema import SearchQuery
from search.enrich import apply_extraction

Q = SearchQuery(query="q")
WEEK = SearchQuery(query="q", time_range="week")
NOW = datetime.now(timezone.utc)
OLD = NOW - timedelta(days=400)


def ok(url, text="page text", **kw):
    return Extracted(url=url, final_url=url, text=text, **kw)


def failed(url, error="Timeout error"):
    return Extracted(url=url, error=error)


def test_success_sets_text():
    c = cand("https://a.com/", 1, snippet="snip")
    [out] = apply_extraction([c], [ok(c.url, title="Page title")], Q)
    assert out.text == "page text" and out.extract_error is None
    assert out.title == "ddg#1" and out.url == "https://a.com/"      # the engine's title and url are kept


def test_missing_title_filled_from_page():
    c = cand("https://a.com/", 1, title="")
    [out] = apply_extraction([c], [ok(c.url, title="Page title")], Q)
    assert out.title == "Page title"


def test_failure_keeps_candidate_with_snippet():
    c = cand("https://a.com/", 1, snippet="snip")
    [out] = apply_extraction([c], [failed(c.url)], Q)
    assert out.text == "snip" and out.extract_error == "Timeout error"


def test_failure_without_snippet():
    c = cand("https://a.com/", 1)
    [out] = apply_extraction([c], [failed(c.url)], Q)
    assert out.text is None and out.extract_error


def test_engine_date_wins():
    c = cand("https://a.com/", 1, published_at=NOW)
    [out] = apply_extraction([c], [ok(c.url, published_at=OLD)], WEEK)
    assert out.published_at == NOW                                    # not replaced, so not dropped


def test_page_date_fills_gap():
    c = cand("https://a.com/", 1)
    [out] = apply_extraction([c], [ok(c.url, published_at=NOW - timedelta(days=2))], WEEK)
    assert out.published_at == NOW - timedelta(days=2)


def test_page_date_out_of_window_dropped():
    a, b = cand("https://a.com/", 1), cand("https://b.com/", 2)
    out = apply_extraction([a, b], [ok(a.url, published_at=OLD), ok(b.url)], WEEK)
    assert [c.url for c in out] == ["https://b.com/"]


def test_no_date_filter_drops_nothing():
    c = cand("https://a.com/", 1)
    assert len(apply_extraction([c], [ok(c.url, published_at=OLD)], Q)) == 1


def test_order_preserved():
    cs = [cand(f"https://s{n}.com/", n + 1) for n in range(5)]
    es = [ok(c.url) if n % 2 else failed(c.url) for n, c in enumerate(cs)]
    assert [c.url for c in apply_extraction(cs, es, Q)] == [c.url for c in cs]


# --- exact_match ------------------------------------------------------------------

def exact(query, *pages, exact_match=True):
    """pages: (title, text) pairs; text None means extraction failed and the snippet is used."""
    cs = [cand(f"https://s{n}.com/", n + 1, title=t, snippet=f"snippet {n}") for n, (t, _) in enumerate(pages)]
    es = [ok(c.url, text) if text is not None else failed(c.url) for c, (_, text) in zip(cs, pages)]
    return [int(c.url[9]) for c in apply_extraction(cs, es, SearchQuery(query=query, exact_match=exact_match))]


def test_no_quotes_or_flag_off_drops_nothing():
    assert exact("task group", ("t", "nothing here")) == [0]
    assert exact('"task group"', ("t", "nothing here"), exact_match=False) == [0]


def test_case_and_punctuation_ignored():
    assert exact('"task group"', ("t", "Using a Task-Group:\nhere"), ("t", "TASK\n  GROUP")) == [0, 1]


def test_whole_words_only():
    assert exact('"task group"', ("t", "the TaskGroup class"), ("t", "multitask groups")) == []


def test_every_phrase_required():
    assert exact('"task group" "python 3.11"', ("t", "a task group in Python 3.11"), ("t", "a task group")) == [0]


def test_phrase_in_title_counts():
    assert exact('"task group"', ("Python Task Group Tutorial", "unrelated body")) == [0]


def test_failed_extraction_checks_snippet():
    # snippets are "snippet 0", "snippet 1"
    assert exact('"snippet 1"', ("t", None), ("t", None)) == [1]


def test_punctuation_only_phrase_ignored():
    assert exact('"!!" asyncio', ("t", "anything")) == [0]


# --- step 6: markdown and favicon are carried over ------------------------------------

def test_markdown_and_favicon_copied_on_success():
    c = cand("https://a.com/", 1)
    [out] = apply_extraction([c], [ok(c.url, markdown="# page", favicon="https://a.com/i.png")], Q)
    assert out.markdown == "# page" and out.favicon == "https://a.com/i.png"


def test_failed_page_keeps_its_favicon_but_no_markdown():
    c = cand("https://a.com/", 1, snippet="snip")
    e = Extracted(url=c.url, error="Timeout error", markdown="stale", favicon="https://a.com/favicon.ico")
    [out] = apply_extraction([c], [e], Q)
    assert out.markdown is None and out.favicon == "https://a.com/favicon.ico"

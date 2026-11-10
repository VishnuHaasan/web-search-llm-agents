from datetime import datetime, timezone, timedelta, date

import pytest

from conftest import cand
from models.schema import SearchQuery
from search.filters import Filters


def _passes(c, q):
    start, end = Filters.date_window(q)
    return Filters.passes_filters(c, q, start, end, set(q.include_domains), set(q.exclude_domains))


@pytest.mark.parametrize("url, expected", [
    ("https://www.Example.com/a/", "example.com/a"),
    ("http://example.com/a", "example.com/a"),
    ("https://example.com/a", "example.com/a"),
])
def test_canonicalize_url(url, expected):
    assert Filters.canonicalize_url(url) == expected


def test_canonicalize_url_raises_on_broken_url():
    # the reason is_fetchable must run before canonicalize_url
    with pytest.raises(ValueError):
        Filters.canonicalize_url("http://[::1")


@pytest.mark.parametrize("url, expected", [
    ("https://docs.python.org/3/", True),
    ("https://python.org/", True),
    ("https://www.python.org/", True),
    ("https://evilpython.org/", False),  # whole labels, not a substring
    ("https://python.org.evil.com/", False),
])
def test_matches_domains(url, expected):
    assert Filters.matches_domains(url, {"python.org"}) is expected


@pytest.mark.parametrize("url, expected", [
    ("https://docs.python.org/x", "python.org"),
    ("https://www.python.org/x", "python.org"),
    ("https://user1.github.io/x", "user1.github.io"),  # github.io is a private public suffix
])
def test_domain_of(url, expected):
    assert Filters.domain_of(url) == expected


@pytest.mark.parametrize("url, expected", [
    ("https://docs.python.org/3/", True),
    ("https://www.youtube.com/watch?v=x", False),   # no-extract list
    ("https://reddit.com/r/python", False),          # no-extract list
    ("http://127.0.0.1/admin", False),               # bare IP has no suffix
    ("http://localhost/x", False),
    ("https://example.com/logo.png", False),         # courlan rejects media files
])
def test_is_fetchable(url, expected):
    assert Filters.is_fetchable(url) is expected


def test_passes_filters_no_filters():
    assert _passes(cand("https://docs.python.org/3/", 1), SearchQuery(query="x"))


def test_passes_filters_blocklist():
    assert not _passes(cand("https://code-examples.net/a", 1), SearchQuery(query="x"))
    assert not _passes(cand("https://sub.spam-site.xyz/a", 1), SearchQuery(query="x"))


def test_allow_list_overrides_blocklist():
    assert _passes(cand("https://allowed-spam.net/a", 1), SearchQuery(query="x"))


def test_passes_filters_exclude():
    q = SearchQuery(query="x", exclude_domains=["python.org"])
    assert not _passes(cand("https://docs.python.org/3/", 1), q)
    assert _passes(cand("https://realpython.com/a", 1), q)


def test_passes_filters_include():
    q = SearchQuery(query="x", include_domains=["python.org"])
    assert _passes(cand("https://docs.python.org/3/", 1), q)
    assert not _passes(cand("https://realpython.com/a", 1), q)


def test_date_window_time_range():
    start, end = Filters.date_window(SearchQuery(query="x", time_range="week"))
    assert end is None
    assert abs((datetime.now(timezone.utc) - timedelta(days=7)) - start) < timedelta(seconds=5)


def test_date_window_end_is_exclusive_next_day():
    start, end = Filters.date_window(SearchQuery(query="x", start_date=date(2025, 1, 1), end_date=date(2025, 1, 31)))
    assert start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2025, 2, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("published, expected", [
    (datetime(2025, 1, 15, tzinfo=timezone.utc), True),
    (datetime(2025, 1, 31, 23, 59, tzinfo=timezone.utc), True),   # end_date is inclusive
    (datetime(2025, 2, 1, tzinfo=timezone.utc), False),
    (datetime(2024, 12, 31, tzinfo=timezone.utc), False),
])
def test_passes_filters_date_window(published, expected):
    q = SearchQuery(query="x", start_date=date(2025, 1, 1), end_date=date(2025, 1, 31))
    assert _passes(cand("https://example.com/a", 1, published_at=published), q) is expected


def test_undated_results_with_a_date_filter():
    news = SearchQuery(query="x", topic="news", time_range="week")
    general = SearchQuery(query="x", time_range="week")
    assert not _passes(cand("https://example.com/a", 1), news)
    assert _passes(cand("https://example.com/a", 1), general)


def test_filters_need_load(monkeypatch):
    monkeypatch.setattr(Filters, "_blocklist", None)
    with pytest.raises(RuntimeError):
        Filters.is_fetchable("https://example.com/")

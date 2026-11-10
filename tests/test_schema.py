from datetime import date

import pytest
from pydantic import ValidationError

from models.schema import SearchQuery
from search.schema import Answer, WebResult


@pytest.mark.parametrize("raw, expected", [
    ("python.org", "python.org"),
    ("https://www.X.com/path", "x.com"),
    ("  www.docs.python.org ", "docs.python.org"),
])
def test_domain_normalized(raw, expected):
    assert SearchQuery(query="x", include_domains=[raw]).include_domains == [expected]


def test_domains_deduplicated_in_order():
    q = SearchQuery(query="x", exclude_domains=["b.com", "a.com", "www.b.com"])
    assert q.exclude_domains == ["b.com", "a.com"]


@pytest.mark.parametrize("bad", ["x.com OR evil", "site:x.com", "localhost", "-x.com", ""])
def test_bad_domain_rejected(bad):
    with pytest.raises(ValidationError):
        SearchQuery(query="x", include_domains=[bad])


def test_time_range_and_dates_together_rejected():
    with pytest.raises(ValidationError):
        SearchQuery(query="x", time_range="week", start_date=date(2025, 1, 1))


def test_start_after_end_rejected():
    with pytest.raises(ValidationError):
        SearchQuery(query="x", start_date=date(2025, 2, 1), end_date=date(2025, 1, 1))


@pytest.mark.parametrize("field, value", [("query", ""), ("max_results", 0), ("max_results", 21)])
def test_limits(field, value):
    with pytest.raises(ValidationError):
        SearchQuery(**{"query": "x", field: value})


def test_unknown_fields_ignored():
    # Tavily clients may send api_key in the body
    assert SearchQuery(query="x", api_key="tvly-123").query == "x"


@pytest.mark.parametrize("value", [[], None])
def test_empty_table_becomes_dict(value):
    # 4get sends table: [] or null when there is no table
    assert WebResult(title="t", url="https://x.com", table=value).table == {}
    assert Answer(title="t", description=[], table=value, sublink=value).sublink == {}

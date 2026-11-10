import os
import subprocess
import sys
from pathlib import Path

from models.schema import SearchQuery
from search.operators import operator_queries


def _q(**kw) -> SearchQuery:
    return SearchQuery(query="asyncio", **kw)


def test_no_domains():
    assert operator_queries(_q()) == ["asyncio"]


def test_up_to_three_includes_one_query_each():
    assert operator_queries(_q(include_domains=["a.com", "b.com", "c.com"])) == [
        "asyncio site:a.com", "asyncio site:b.com", "asyncio site:c.com"]


def test_more_than_three_includes_one_or_query():
    assert operator_queries(_q(include_domains=["a.com", "b.com", "c.com", "d.com"])) == [
        "asyncio (site:a.com OR site:b.com OR site:c.com OR site:d.com)"]


def test_excludes_capped_at_five():
    q = _q(exclude_domains=[f"x{i}.com" for i in range(7)])
    assert operator_queries(q) == ["asyncio -site:x0.com -site:x1.com -site:x2.com -site:x3.com -site:x4.com"]


def test_excludes_added_to_every_include_query():
    q = _q(include_domains=["a.com", "b.com"], exclude_domains=["z.com"])
    assert operator_queries(q) == ["asyncio site:a.com -site:z.com", "asyncio site:b.com -site:z.com"]


def test_included_and_excluded_domain_dropped_from_includes():
    q = _q(include_domains=["a.com", "b.com"], exclude_domains=["a.com"])
    assert operator_queries(q) == ["asyncio site:b.com -site:a.com"]


def test_all_includes_excluded_gives_no_queries():
    # otherwise it would search the whole web
    assert operator_queries(_q(include_domains=["a.com"], exclude_domains=["a.com"])) == []


def test_quotes_passed_through():
    # exact_match is enforced on page text in stage 2, the query is not changed
    q = SearchQuery(query='"asyncio" tutorial', exact_match=True)
    assert operator_queries(q) == ['"asyncio" tutorial']


def test_order_stable_across_hash_seeds():
    # sets would make the order depend on PYTHONHASHSEED
    code = ("from models.schema import SearchQuery; from search.operators import operator_queries; "
            "print(operator_queries(SearchQuery(query='q', include_domains=['a.com','b.com','c.com'], exclude_domains=['x.com','y.com'])))")
    src = str(Path(__file__).resolve().parents[1] / "src")
    outputs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1

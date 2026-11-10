import time

import pytest

from models.schema import SearchQuery
from search.fourget.scraper import FourGetScraperPool
from utils.errors import FourGetError


@pytest.fixture
def clock(monkeypatch):
    # the pool reads time.monotonic(); tests move this clock instead of waiting
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def _cooldown(pool, scraper, clock):
    return pool.scraper_states[scraper].cooldown_until - clock[0]


def test_blocked_doubles_up_to_one_hour(clock):
    pool = FourGetScraperPool()
    waits = []
    for _ in range(6):
        pool.report_failure("brave", FourGetError.BLOCKED)
        waits.append(_cooldown(pool, "brave", clock))
    assert waits == [300, 600, 1200, 2400, 3600, 3600]


def test_transient_doubles_up_to_five_minutes(clock):
    pool = FourGetScraperPool()
    waits = []
    for _ in range(6):
        pool.report_failure("ddg", FourGetError.TRANSIENT)
        waits.append(_cooldown(pool, "ddg", clock))
    assert waits == [20, 40, 80, 160, 300, 300]


def test_success_resets(clock):
    pool = FourGetScraperPool()
    pool.report_failure("ddg", FourGetError.BLOCKED)
    pool.report_failure("ddg", FourGetError.BLOCKED)
    pool.report_success("ddg")
    assert pool.scraper_states["ddg"].fail_count == 0
    pool.report_failure("ddg", FourGetError.BLOCKED)
    assert _cooldown(pool, "ddg", clock) == 300


def test_pick_skips_cooling_and_comes_back(clock):
    pool = FourGetScraperPool()
    pool.report_failure("ddg", FourGetError.BLOCKED)
    assert pool.pick(["ddg", "startpage"]) == "startpage"
    clock[0] += 301
    assert pool.pick(["ddg", "startpage"]) == "ddg"


def test_pick_raises_when_all_cooling(clock):
    pool = FourGetScraperPool()
    for s in ("ddg", "startpage"):
        pool.report_failure(s, FourGetError.TRANSIENT)
    with pytest.raises(FourGetError) as e:
        pool.pick(["ddg", "startpage"])
    assert e.value.kind == FourGetError.BLOCKED


def test_order_by_topic():
    pool = FourGetScraperPool()
    assert pool.order(SearchQuery(query="x"))[0] == "ddg"
    assert "startpage" in pool.order(SearchQuery(query="x"))
    assert "startpage" not in pool.order(SearchQuery(query="x", topic="news"))


def test_order_uses_only_requested_engines():
    pool = FourGetScraperPool()
    assert pool.order(SearchQuery(query="x", engines=["qwant", "bing", "ddg"])) == ["qwant", "ddg"]


def test_order_all_unknown_engines_is_invalid():
    with pytest.raises(FourGetError) as e:
        FourGetScraperPool().order(SearchQuery(query="x", engines=["bing"]))
    assert e.value.kind == FourGetError.INVALID

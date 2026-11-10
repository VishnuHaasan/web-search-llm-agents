import datetime

import pytest

from models.schema import SearchQuery
from search.auto import apply_auto_parameters


def auto(query, **kw):
    q, echoed = apply_auto_parameters(SearchQuery(query=query, auto_parameters=True, **kw))
    return q, echoed


# --- topic ------------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["openai news", "Headlines from Berlin", "headline today", "breaking: quake",
                                   "apple announced a phone", "rust 2.0 announcement", "nvidia press release"])
def test_news_words(query):
    assert auto(query)[0].topic == "news"


@pytest.mark.parametrize("query", ["newsletter templates", "python asyncio", "announce library"])
def test_not_news(query):
    assert auto(query)[0].topic == "general"


# --- time_range ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    ("weather today", "day"), ("events tonight", "day"), ("what is happening right now", "day"),
    ("breaking quake", "day"),
    ("rust release this week", "week"), ("python past week", "week"), ("top stories last week", "week"),
    ("gpu prices this month", "month"), ("past month layoffs", "month"), ("last month inflation", "month"),
    ("best phones this year", "year"), ("past year review", "year"), ("last year results", "year"),
])
def test_time_phrases(query, expected):
    assert auto(query)[0].time_range == expected


def test_first_time_phrase_wins():
    assert auto("today vs this week")[0].time_range == "day"


@pytest.mark.parametrize("query", ["latest python release", "recent rust news", "newest iphone",
                                   "2023 election results", "week numbers in excel", "monthly budget"])
def test_no_time_range(query):
    assert auto(query)[0].time_range is None


# --- search_depth -------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["how do I stop a TaskGroup", "Why is the sky blue", "what is rrf",
                                   "which gpu for llms", "gather vs TaskGroup", "rust versus go",
                                   "compare postgres and mysql", "difference between list and tuple",
                                   "asyncio taskgroup?"])
def test_questions_get_advanced(query):
    assert auto(query)[0].search_depth == "advanced"


@pytest.mark.parametrize("query", ["python asyncio taskgroup", "showhow tutorial", "canvas api"])
def test_keywords_stay_basic(query):
    assert auto(query)[0].search_depth == "basic"


# --- quoted phrases and site: -------------------------------------------------------------

def test_quoted_phrase_ignored():
    q, echoed = auto('"breaking bad" cast')
    assert echoed == {"topic": "general", "time_range": None, "search_depth": "basic"}


def test_site_operator_ignored():
    assert auto("python site:news.ycombinator.com")[0].topic == "general"


# --- the client always wins ------------------------------------------------------------------

def test_sent_topic_kept_even_if_default():
    q, echoed = auto("news today", topic="general")
    assert q.topic == "general" and q.time_range == "day"


def test_sent_depth_kept():
    assert auto("how does rrf work", search_depth="basic")[0].search_depth == "basic"


def test_sent_time_range_kept():
    assert auto("news today", time_range="month")[0].time_range == "month"


@pytest.mark.parametrize("field", ["start_date", "end_date"])
def test_dates_block_time_range(field):
    q, _ = auto("news today", **{field: datetime.date(2026, 1, 1)})
    assert q.topic == "news" and q.time_range is None


# --- echo -----------------------------------------------------------------------------------

def test_echo_has_final_values():
    q, echoed = auto("openai news today")
    assert echoed == {"topic": "news", "time_range": "day", "search_depth": "basic"}


def test_echo_includes_client_values():
    _, echoed = auto("python", topic="technology", search_depth="advanced", time_range="year")
    assert echoed == {"topic": "technology", "time_range": "year", "search_depth": "advanced"}


def test_other_fields_untouched():
    q, _ = auto("how to news today", max_results=7, include_answer=True)
    assert q.max_results == 7 and q.include_answer is True and q.query == "how to news today"

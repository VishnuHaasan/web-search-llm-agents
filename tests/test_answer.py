from datetime import date, datetime, timezone

import pytest

from conftest import cand
from llm.answer import _build_prompt, _format_source, _format_sources, _parse_questions, generate_answer, generate_follow_ups
from llm.client import LLMError
from models.schema import SearchQuery


def src(n, content="text", **kw):
    return cand(f"https://www.s{n}.com/page", n, content=content, **kw)


class FakeLLM:
    def __init__(self, reply="The answer.", error=None):
        self.reply, self.error, self.calls = reply, error, []

    async def chat(self, system, user, max_tokens):
        self.calls.append((system, user, max_tokens))
        if self.error:
            raise self.error
        return self.reply


# --- _format_source ---------------------------------------------------------------------

def test_source_with_everything():
    c = cand("https://www.geeksforgeeks.org/python/x", 1, title="TaskGroups", content="body",
             published_at=datetime(2024, 3, 12, tzinfo=timezone.utc))
    assert _format_source(1, c) == "[1] TaskGroups (geeksforgeeks.org, 2024-03-12)\nbody"


def test_source_without_date_keeps_subdomain():
    assert _format_source(2, cand("https://docs.python.org/3/", 1, title="Docs", content="b")) == "[2] Docs (docs.python.org)\nb"


def test_source_falls_back_to_snippet():
    c = cand("https://a.com/", 1, title="A", content="", snippet="  engine snippet ")
    assert _format_source(3, c) == "[3] A (a.com)\nengine snippet"


def test_source_without_title():
    assert _format_source(4, cand("https://a.com/", 1, title="", content="b")) == "[4] (a.com)\nb"


@pytest.mark.parametrize("content, snippet", [("", None), (None, ""), ("   ", "  \n ")])
def test_source_without_text_is_skipped(content, snippet):
    assert _format_source(1, cand("https://a.com/", 1, content=content, snippet=snippet)) is None


# --- _format_sources ----------------------------------------------------------------------

def test_numbering_has_no_gaps_and_looks_past_empty_ones():
    cs = [src(1), src(2, content=""), src(3), src(4)]
    out = _format_sources(cs, 3)
    assert [b.split(" ")[0] for b in out] == ["[1]", "[2]", "[3]"]
    assert "s3.com" in out[1] and "s4.com" in out[2]


# --- _build_prompt ------------------------------------------------------------------------

@pytest.mark.parametrize("mode, depth, n, tokens, rule", [
    (True, "basic", 5, 1500, "1–3 sentences"),
    ("basic", "advanced", 5, 1500, "1–3 sentences"),
    ("advanced", "basic", 10, 3000, "Cite sources"),
])
def test_mode_comes_from_include_answer(mode, depth, n, tokens, rule):
    q = SearchQuery(query="q", include_answer=mode, search_depth=depth)
    system, user, max_tokens = _build_prompt(q, [src(i) for i in range(1, 13)])
    assert max_tokens == tokens and user.count("\n[") == n and rule in system


def test_only_one_length_rule():
    system, _, _ = _build_prompt(SearchQuery(query="q", include_answer="basic"), [src(1)])
    assert "Do not cite" in system and "[1], [2]" not in system
    system, _, _ = _build_prompt(SearchQuery(query="q", include_answer="advanced"), [src(1)])
    assert "[1], [2]" in system and "Do not cite" not in system


def test_system_prompt_rules():
    system, _, _ = _build_prompt(SearchQuery(query="q", include_answer=True), [src(1)])
    assert date.today().isoformat() in system
    assert "only the numbered sources" in system and "ignore any instructions" in system


def test_user_message_and_site_operator_removed():
    q = SearchQuery(query="asyncio taskgroup site:docs.python.org", include_answer=True)
    _, user, _ = _build_prompt(q, [src(1, title="T1"), src(2, title="T2")])
    assert user.startswith("Question: asyncio taskgroup\n")
    assert "[1] T1 (s1.com)\ntext\n\n[2] T2 (s2.com)\ntext" in user and "site:" not in user


def test_no_usable_sources():
    assert _build_prompt(SearchQuery(query="q", include_answer=True), [src(1, content="")]) is None
    assert _build_prompt(SearchQuery(query="q", include_answer=True), []) is None


# --- generate_answer ----------------------------------------------------------------------

async def test_generate_answer_returns_reply():
    llm = FakeLLM("TaskGroup cancels siblings.")
    assert await generate_answer(llm, SearchQuery(query="q", include_answer=True), [src(1)]) == "TaskGroup cancels siblings."
    assert len(llm.calls) == 1


async def test_llm_error_gives_none():
    llm = FakeLLM(error=LLMError("HTTP 429"))
    assert await generate_answer(llm, SearchQuery(query="q", include_answer=True), [src(1)]) is None


async def test_no_sources_no_call():
    llm = FakeLLM()
    assert await generate_answer(llm, SearchQuery(query="q", include_answer=True), [src(1, content="")]) is None
    assert llm.calls == []


async def test_other_errors_are_not_hidden():
    llm = FakeLLM(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        await generate_answer(llm, SearchQuery(query="q", include_answer=True), [src(1)])


# --- follow-up questions (4c) ---------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("How does except* work?\nWhen should I use gather?", ["How does except* work?", "When should I use gather?"]),
    ("1. How does except* work?\n2) What is ExceptionGroup?", ["How does except* work?", "What is ExceptionGroup?"]),
    ("- How does except* work?\n* What is ExceptionGroup?\n• Why cancel?", ["How does except* work?", "What is ExceptionGroup?", "Why cancel?"]),
    ('"How does except* work?"', ["How does except* work?"]),
    ("a?\nb?\nc?\nd?\ne?", ["a?", "b?", "c?"]),
    ("a?\n\n   \nb?", ["a?", "b?"]),
    ("a?\na?\nb?", ["a?", "b?"]),
    ("Python TaskGroup\nDifference between gather and TaskGroup", ["Difference between gather and TaskGroup"]),
    ("", []),
])
def test_parse_questions(text, expected):
    assert _parse_questions(text, "python taskgroup") == expected


async def test_follow_ups_prompt_and_result():
    llm = FakeLLM("1. How does except* work?\n2. What is an ExceptionGroup?")
    q = SearchQuery(query="asyncio taskgroup site:docs.python.org", include_follow_up_questions=True)
    out = await generate_follow_ups(llm, q, [src(i, title=f"T{i}") for i in range(1, 8)])
    assert out == ["How does except* work?", "What is an ExceptionGroup?"]
    [(system, user, max_tokens)] = llm.calls
    assert max_tokens == 300 and system.startswith("Suggest 3 follow-up questions")
    assert user.startswith("Question: asyncio taskgroup\n") and user.count("\n[") == 5


async def test_follow_ups_llm_error_gives_empty_list():
    llm = FakeLLM(error=LLMError("HTTP 500"))
    assert await generate_follow_ups(llm, SearchQuery(query="q"), [src(1)]) == []


async def test_follow_ups_no_sources_no_call():
    llm = FakeLLM()
    assert await generate_follow_ups(llm, SearchQuery(query="q"), [src(1, content="")]) == []
    assert llm.calls == []


async def test_follow_ups_other_errors_are_not_hidden():
    with pytest.raises(RuntimeError):
        await generate_follow_ups(FakeLLM(error=RuntimeError("bug")), SearchQuery(query="q"), [src(1)])

import asyncio
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import app
from conftest import load_fixture, make_provider
from fetch.extract import Extracted, Extractor
from fetch.safe_fetch import make_client
from llm.client import LLMError
from search.filters import Filters


class FakeExtractor:
    """Stands in for the real one so tests never fetch pages. pages: url -> Extracted (default: a failure)."""

    def __init__(self, pages=None):
        self.pages = pages or {}
        self.urls: list[str] = []

    async def extract_many(self, urls, remaining=None, markdown=False):
        self.urls, self.remaining, self.markdown = list(urls), remaining, markdown
        return [self.pages.get(u, Extracted(url=u, error="not fetched in tests")) for u in urls]


@pytest.fixture
def api(filters_dir):
    """The app with a fake 4get. Swap the handler with api.use(handler)."""
    try:
        with TestClient(app) as client:          # the lifespan only runs inside `with`
            Filters.load(filters_dir)            # the lifespan loaded the real data/; put the test lists back

            def use(handler):
                app.state.providers = [make_provider(handler)]

            def extract(pages=None):
                app.state.extractor = FakeExtractor(pages)
                return app.state.extractor

            def llm(fake=None):
                app.state.llm_client = fake
                return fake

            client.use, client.extract, client.llm = use, extract, llm
            extract()
            llm()                                # never call the real gateway from .env
            use(lambda r: httpx.Response(200, json=load_fixture("web_ddg.json")))
            yield client
    finally:
        Filters.load(filters_dir)                # also when the lifespan itself fails


def test_search_ok(api):
    r = api.post("/search", json={"query": "asyncio", "max_results": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"query", "answer", "images", "results", "response_time", "request_id"}
    assert len(body["results"]) == 3
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True) and all(0 < s <= 1 for s in scores)   # top is not forced to 1.0
    assert all(x["url"].startswith("http") and isinstance(x["content"], str) for x in body["results"])


def test_request_ids_differ(api):
    ids = {api.post("/search", json={"query": "asyncio"}).json()["request_id"] for _ in range(2)}
    assert len(ids) == 2


@pytest.mark.parametrize("body", [
    {"query": ""},
    {"query": "x", "max_results": 50},
    {"query": "x", "include_domains": ["x.com OR evil"]},
    {"query": "x", "time_range": "week", "start_date": "2025-01-01"},
    {},
])
def test_bad_input_is_422(api, body):
    assert api.post("/search", json=body).status_code == 422


def test_4get_down_is_502_with_reasons(api):
    def down(request):
        raise httpx.ConnectError("refused")
    api.use(down)
    r = api.post("/search", json={"query": "asyncio"})
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"][0]["message"]


def test_nothing_to_search_is_empty_200(api):
    r = api.post("/search", json={"query": "x", "include_domains": ["a.com"], "exclude_domains": ["a.com"]})
    assert r.status_code == 200 and r.json()["results"] == []


def test_images_returned_as_urls(api):
    def handler(request):
        name = "images_ddg.json" if request.url.path.endswith("/images") else "web_ddg.json"
        return httpx.Response(200, json=load_fixture(name))
    api.use(handler)
    images = api.post("/search", json={"query": "python logo", "include_images": True}).json()["images"]
    assert len(images) == 10 and all(isinstance(i, str) for i in images)


def test_answer_from_answer_box(api):
    api.use(lambda r: httpx.Response(200, json=load_fixture("web_answer_ddg.json")))
    body = api.post("/search", json={"query": "python programming language", "include_answer": True}).json()
    assert body["answer"].startswith("Python is a high-level")


def test_news_dates(api):
    api.use(lambda r: httpx.Response(200, json=load_fixture("news_ddg.json")))
    results = api.post("/search", json={"query": "python", "topic": "news"}).json()["results"]
    assert results and all(x["published_date"] for x in results)


# --- extraction in the route ------------------------------------------------------

def test_every_candidate_is_extracted_then_trimmed(api):
    fake = api.extract()
    results = api.post("/search", json={"query": "asyncio", "max_results": 3}).json()["results"]
    assert len(fake.urls) == 6 and len(results) == 3                  # candidate_target = 2 * 3
    assert {x["url"] for x in results} <= set(fake.urls)              # order comes from the ranker now


def test_raw_content_only_when_asked(api):
    fake = api.extract()
    api.post("/search", json={"query": "asyncio", "max_results": 3})
    api.extract({u: Extracted(url=u, text=f"full page text for {u}") for u in fake.urls})
    plain = api.post("/search", json={"query": "asyncio", "max_results": 3}).json()["results"]
    raw = api.post("/search", json={"query": "asyncio", "max_results": 3, "include_raw_content": True}).json()["results"]
    assert all(x["raw_content"] is None for x in plain)
    assert all(x["raw_content"] == f"full page text for {x['url']}" for x in raw)


def test_spares_fill_in_for_dropped_pages(api):
    body = {"query": "asyncio", "max_results": 3, "time_range": "week"}
    before = [x["url"] for x in api.post("/search", json=body).json()["results"]]
    old = datetime.now(timezone.utc) - timedelta(days=400)
    api.extract({before[0]: Extracted(url=before[0], text="x", published_at=old)})
    after = [x["url"] for x in api.post("/search", json=body).json()["results"]]
    assert before[0] not in after and len(after) == 3


def test_no_snippet_and_failed_extraction(api):
    page = load_fixture("web_ddg.json")
    page["web"][0]["description"] = ""
    api.use(lambda r: httpx.Response(200, json=page))
    r = api.post("/search", json={"query": "asyncio", "max_results": 3})
    assert r.status_code == 200 and isinstance(r.json()["results"][0]["content"], str)


def test_exact_match_spares_fill_in(api):
    body = {"query": '"the phrase" asyncio', "max_results": 3, "exact_match": True}
    fake = api.extract()
    api.post("/search", json=body)
    urls = fake.urls                                                  # every candidate the route extracts
    api.extract({u: Extracted(url=u, text="nope" if u == urls[0] else "the PHRASE, here") for u in urls})
    results = [x["url"] for x in api.post("/search", json=body).json()["results"]]
    assert results == urls[1:4]


def test_slow_search_still_gets_min_extract_time(api, monkeypatch):
    fake = api.extract()
    api.post("/search", json={"query": "asyncio"})
    assert 14 < fake.remaining <= 15                                  # a fast search leaves almost all of it
    monkeypatch.setattr(api.app.state, "max_request_budget", 0.0)
    api.post("/search", json={"query": "asyncio"})
    assert fake.remaining == api.app.state.min_extract_time


def md_pages(urls):
    return {u: Extracted(url=u, text=f"plain {u}", markdown=f"# md {u}", favicon=f"{u}icon.png") for u in urls}


@pytest.mark.parametrize("mode, want_md, expected", [
    (True, True, "# md {u}"),
    ("markdown", True, "# md {u}"),
    ("text", False, "plain {u}"),
    (False, False, None),
])
def test_raw_content_format(api, mode, want_md, expected):
    fake = api.extract()
    api.post("/search", json={"query": "asyncio", "max_results": 3})
    fake = api.extract(md_pages(fake.urls))
    results = api.post("/search", json={"query": "asyncio", "max_results": 3,
                                        "include_raw_content": mode}).json()["results"]
    assert fake.markdown is want_md
    assert all(x["raw_content"] == (expected and expected.format(u=x["url"])) for x in results)


def test_raw_content_markdown_falls_back_to_plain(api):
    fake = api.extract()
    api.post("/search", json={"query": "asyncio", "max_results": 3})
    api.extract({u: Extracted(url=u, text=f"plain {u}") for u in fake.urls})
    results = api.post("/search", json={"query": "asyncio", "max_results": 3,
                                        "include_raw_content": "markdown"}).json()["results"]
    assert all(x["raw_content"] == f"plain {x['url']}" for x in results)


def test_favicon_only_when_asked(api):
    fake = api.extract()
    api.post("/search", json={"query": "asyncio", "max_results": 3})
    api.extract(md_pages(fake.urls))
    off = api.post("/search", json={"query": "asyncio", "max_results": 3}).json()["results"]
    on = api.post("/search", json={"query": "asyncio", "max_results": 3, "include_favicon": True}).json()["results"]
    assert all(x["favicon"] is None for x in off)
    assert all(x["favicon"] == f"{x['url']}icon.png" for x in on)


# --- POST /extract ----------------------------------------------------------------

def page(url, text="page text"):
    return Extracted(url=url, final_url=url, text=text)


def test_extract_splits_results_and_failures(api):
    api.extract({"https://a.com/": page("https://a.com/"), "https://c.com/": page("https://c.com/", "c text")})
    r = api.post("/extract", json={"urls": ["https://a.com/", "https://b.com/", "https://c.com/"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"results", "failed_results", "response_time", "request_id"}
    assert [(x["url"], x["raw_content"]) for x in body["results"]] == [("https://a.com/", "page text"), ("https://c.com/", "c text")]
    assert body["failed_results"] == [{"url": "https://b.com/", "error": "not fetched in tests"}]


def test_extract_single_string_url(api):
    fake = api.extract()
    assert api.post("/extract", json={"urls": "https://a.com/"}).status_code == 200
    assert fake.urls == ["https://a.com/"]


def test_extract_duplicates_fetched_once(api):
    fake = api.extract()
    api.post("/extract", json={"urls": ["https://a.com/", "https://b.com/", "https://a.com/"]})
    assert fake.urls == ["https://a.com/", "https://b.com/"]


@pytest.mark.parametrize("body, timeout", [
    ({}, 10.0),
    ({"extract_depth": "advanced"}, 30.0),
    ({"extract_depth": "advanced", "timeout": 4.5}, 4.5),
])
def test_extract_timeout(api, body, timeout):
    fake = api.extract()
    assert api.post("/extract", json={"urls": ["https://a.com/"], **body}).status_code == 200
    assert fake.remaining == timeout


@pytest.mark.parametrize("body", [
    {"urls": []},
    {"urls": [f"https://s{n}.com/" for n in range(21)]},
    {"urls": ["https://a.com/"], "timeout": 0.5},
    {"urls": ["https://a.com/"], "timeout": 61},
    {"urls": ["https://a.com/"], "extract_depth": "deep"},
    {},
])
def test_extract_bad_input_is_422(api, body):
    assert api.post("/extract", json=body).status_code == 422


def test_extract_unimplemented_options_accepted(api):
    r = api.post("/extract", json={"urls": ["https://a.com/"], "format": "text", "include_images": True,
                                   "include_favicon": True, "query": "q", "chunks_per_source": 2, "include_usage": True})
    assert r.status_code == 200


def test_extract_all_failed_is_still_200(api):
    r = api.post("/extract", json={"urls": ["https://a.com/", "https://b.com/"]})
    assert r.status_code == 200 and r.json()["results"] == [] and len(r.json()["failed_results"]) == 2


def test_extract_blocks_private_urls(api):
    # the real Extractor and safe_get; every one of these is refused before any connection
    api.app.state.extractor = Extractor(make_client(1, 1), page_timeout=2, min_text_filter=50, concurrency=4)
    urls = ["http://127.0.0.1/", "http://10.0.0.1/", "http://[::1]/", "file:///etc/passwd",
            "https://example.com:8080/", "https://user:pw@example.com/"]
    body = api.post("/extract", json={"urls": urls}).json()
    assert body["results"] == []
    assert [f["url"] for f in body["failed_results"]] == urls


def test_content_and_order_come_from_the_page_text(api):
    # only one page contains the query word, so BM25 lifts it above the engines' first choice
    body = {"query": "zebracorn", "max_results": 3}
    fake = api.extract()
    api.post("/search", json=body)
    third = fake.urls[2]                                              # engine position 3
    api.extract({third: Extracted(url=third, text="Intro line.\nA zebracorn runs tasks together; the zebracorn cancels the rest.")})
    results = api.post("/search", json=body).json()["results"]
    assert results[0]["url"] == third
    assert "zebracorn" in results[0]["content"]                       # content is the matching chunk, not the snippet
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)


# --- step 4: domain cap after ranking, sorted scores --------------------------------

def site_page(urls):
    """A 4get answer whose web results are exactly these urls, in this order."""
    page = load_fixture("web_ddg.json")
    item = page["web"][0]
    page["web"] = [{**item, "url": u, "title": u, "description": "a result", "date": None} for u in urls]
    page["npt"] = None
    return page


GFG = [f"https://www.geeksforgeeks.org/p{n}" for n in range(1, 6)]
OTHERS = ["https://alpha.com/x", "https://beta.com/x", "https://gamma.com/x"]
ENGINE_ORDER = GFG[:3] + OTHERS + GFG[3:]


def strong_gfg(api):
    api.use(lambda r: httpx.Response(200, json=site_page(ENGINE_ORDER)))
    api.extract({**{u: Extracted(url=u, text="zebracorn zebracorn guide") for u in GFG},
                 **{u: Extracted(url=u, text="unrelated words only") for u in OTHERS}})


def test_domain_cap_survives_ranking(api):
    strong_gfg(api)                                   # ranking alone would put all 5 gfg pages on top
    results = api.post("/search", json={"query": "zebracorn", "max_results": 5}).json()["results"]
    urls = [x["url"] for x in results]
    assert len(urls) == 5 and sum("geeksforgeeks" in u for u in urls) == 3


def test_scores_stay_sorted_when_a_moved_page_is_returned(api):
    strong_gfg(api)
    results = api.post("/search", json={"query": "zebracorn", "max_results": 8}).json()["results"]
    urls, scores = [x["url"] for x in results], [x["score"] for x in results]
    assert urls[-2:] == GFG[3:]                       # the moved pages come last...
    assert scores == sorted(scores, reverse=True)     # ...with scores lowered to keep the order


def test_single_site_search_keeps_ranked_order(api):
    api.use(lambda r: httpx.Response(200, json=site_page(GFG)))
    api.extract({u: Extracted(url=u, text=("zebracorn " * n) + "guide") for n, u in enumerate(GFG, 1)})
    results = api.post("/search", json={"query": "zebracorn", "max_results": 5,
                                        "include_domains": ["geeksforgeeks.org"]}).json()["results"]
    assert len(results) == 5 and all("geeksforgeeks" in x["url"] for x in results)
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)


def test_fewer_candidates_than_max_results(api):
    api.use(lambda r: httpx.Response(200, json=site_page(OTHERS)))
    assert len(api.post("/search", json={"query": "zebracorn", "max_results": 10}).json()["results"]) == 3


# --- /extract: format and favicon ------------------------------------------------------

@pytest.mark.parametrize("body, want_md, raw", [
    ({}, True, "# md https://a.com/"),                       # Tavily default format is markdown
    ({"format": "markdown"}, True, "# md https://a.com/"),
    ({"format": "text"}, False, "plain https://a.com/"),
])
def test_extract_format(api, body, want_md, raw):
    fake = api.extract(md_pages(["https://a.com/"]))
    [r] = api.post("/extract", json={"urls": "https://a.com/", **body}).json()["results"]
    assert fake.markdown is want_md and r["raw_content"] == raw


def test_extract_markdown_falls_back_to_plain(api):
    api.extract({"https://a.com/": page("https://a.com/", "plain only")})
    [r] = api.post("/extract", json={"urls": "https://a.com/"}).json()["results"]
    assert r["raw_content"] == "plain only"


def test_extract_favicon_only_when_asked(api):
    api.extract(md_pages(["https://a.com/"]))
    [off] = api.post("/extract", json={"urls": "https://a.com/"}).json()["results"]
    [on] = api.post("/extract", json={"urls": "https://a.com/", "include_favicon": True}).json()["results"]
    assert off["favicon"] is None and on["favicon"] == "https://a.com/icon.png"


# --- include_answer (stage 4b) ----------------------------------------------------------

class FakeLLM:
    """Answers with `reply`; system prompts starting with "Suggest" get `follow` (or `follow_error`)."""

    def __init__(self, reply="LLM answer.", error=None, follow="Q one?\nQ two?", follow_error=None, delay=0.0):
        self.reply, self.error, self.follow, self.follow_error, self.delay = reply, error, follow, follow_error, delay
        self.calls = []

    async def chat(self, system, user, max_tokens):
        self.calls.append((system, user, max_tokens))
        await asyncio.sleep(self.delay)
        if system.startswith("Suggest"):
            if self.follow_error:
                raise self.follow_error
            return self.follow
        if self.error:
            raise self.error
        return self.reply


def answer_box(api):
    api.use(lambda r: httpx.Response(200, json=load_fixture("web_answer_ddg.json")))


def test_llm_answer_used(api):
    llm = api.llm(FakeLLM("TaskGroup cancels siblings."))
    body = api.post("/search", json={"query": "asyncio", "include_answer": True}).json()
    assert body["answer"] == "TaskGroup cancels siblings." and len(llm.calls) == 1


def test_llm_answer_wins_over_instant_answer(api):
    answer_box(api)
    api.llm(FakeLLM("From the LLM."))
    body = api.post("/search", json={"query": "python programming language", "include_answer": True}).json()
    assert body["answer"] == "From the LLM."


@pytest.mark.parametrize("mode, tokens", [(True, 1500), ("basic", 1500), ("advanced", 3000)])
def test_answer_mode_reaches_the_llm(api, mode, tokens):
    llm = api.llm(FakeLLM())
    api.post("/search", json={"query": "asyncio", "include_answer": mode})
    assert llm.calls[0][2] == tokens


def test_sources_are_the_returned_results(api):
    llm = api.llm(FakeLLM())
    results = api.post("/search", json={"query": "asyncio", "max_results": 3, "include_answer": True}).json()["results"]
    user = llm.calls[0][1]
    assert user.count("\n[") == 3
    assert all(r["title"] in user for r in results)


def test_no_include_answer_no_call(api):
    llm = api.llm(FakeLLM())
    body = api.post("/search", json={"query": "asyncio"}).json()
    assert body["answer"] is None and llm.calls == []


def test_llm_error_falls_back_to_instant_answer(api):
    answer_box(api)
    api.llm(FakeLLM(error=LLMError("HTTP 429")))
    r = api.post("/search", json={"query": "python programming language", "include_answer": True})
    assert r.status_code == 200 and r.json()["answer"].startswith("Python is a high-level")


def test_llm_error_without_instant_answer_is_none(api):
    api.llm(FakeLLM(error=LLMError("timeout")))
    r = api.post("/search", json={"query": "asyncio", "include_answer": True})
    assert r.status_code == 200 and r.json()["answer"] is None and r.json()["results"]


def test_no_llm_configured_keeps_instant_answer(api):
    answer_box(api)
    body = api.post("/search", json={"query": "python programming language", "include_answer": True}).json()
    assert body["answer"].startswith("Python is a high-level")


def test_no_results_no_call(api):
    llm = api.llm(FakeLLM())
    api.use(lambda r: httpx.Response(200, json={"status": "ok", "web": []}))
    body = api.post("/search", json={"query": "asyncio", "include_answer": True}).json()
    assert body["results"] == [] and body["answer"] is None and llm.calls == []



# --- follow_up_questions (stage 4c) -----------------------------------------------------

def test_follow_ups_off_by_default(api):
    llm = api.llm(FakeLLM())
    body = api.post("/search", json={"query": "asyncio"}).json()
    assert body["follow_up_questions"] == [] and llm.calls == []


def test_follow_ups_only(api):
    llm = api.llm(FakeLLM())
    body = api.post("/search", json={"query": "asyncio", "include_follow_up_questions": True}).json()
    assert body["follow_up_questions"] == ["Q one?", "Q two?"] and body["answer"] is None
    assert len(llm.calls) == 1


def test_answer_and_follow_ups_run_together(api):
    llm = api.llm(FakeLLM(delay=0.5))
    t = time.perf_counter()
    body = api.post("/search", json={"query": "asyncio", "include_answer": True,
                                     "include_follow_up_questions": True}).json()
    assert body["answer"] == "LLM answer." and body["follow_up_questions"] == ["Q one?", "Q two?"]
    assert len(llm.calls) == 2 and time.perf_counter() - t < 0.9          # one delay, not two


def test_follow_up_failure_keeps_the_answer(api):
    api.llm(FakeLLM(follow_error=LLMError("HTTP 500")))
    r = api.post("/search", json={"query": "asyncio", "include_answer": True, "include_follow_up_questions": True})
    assert r.status_code == 200 and r.json()["answer"] == "LLM answer." and r.json()["follow_up_questions"] == []


def test_answer_failure_keeps_the_follow_ups(api):
    api.llm(FakeLLM(error=LLMError("timeout")))
    body = api.post("/search", json={"query": "asyncio", "include_answer": True,
                                     "include_follow_up_questions": True}).json()
    assert body["answer"] is None and body["follow_up_questions"] == ["Q one?", "Q two?"]


def test_follow_ups_without_llm(api):
    body = api.post("/search", json={"query": "asyncio", "include_follow_up_questions": True}).json()
    assert body["follow_up_questions"] == []



# --- auto_parameters (stage 4e) -----------------------------------------------------------

def test_auto_parameters_switch_to_news_endpoint(api):
    paths = []

    def handler(r):
        paths.append(r.url.path)
        return httpx.Response(200, json=load_fixture("news_ddg.json" if r.url.path.endswith("/news") else "web_ddg.json"))

    api.use(handler)
    body = api.post("/search", json={"query": "python news", "auto_parameters": True}).json()
    assert paths and all(p.endswith("/news") for p in paths)
    assert body["auto_parameters"] == {"topic": "news", "time_range": None, "search_depth": "basic"}


def test_auto_parameters_off_changes_nothing(api):
    paths = []

    def handler(r):
        paths.append(r.url.path)
        return httpx.Response(200, json=load_fixture("web_ddg.json"))

    api.use(handler)
    body = api.post("/search", json={"query": "python news today"}).json()
    assert body["auto_parameters"] is None and all(p.endswith("/web") for p in paths)


def test_auto_parameters_respects_explicit_topic(api):
    paths = []

    def handler(r):
        paths.append(r.url.path)
        return httpx.Response(200, json=load_fixture("web_ddg.json"))

    api.use(handler)
    body = api.post("/search", json={"query": "python news", "topic": "general", "auto_parameters": True}).json()
    assert all(p.endswith("/web") for p in paths) and body["auto_parameters"]["topic"] == "general"



# --- include_image_description (stage 4d) -------------------------------------------------

def image_api(api, mutate=None):
    def handler(request):
        if request.url.path.endswith("/images"):
            data = load_fixture("images_ddg.json")
            if mutate:
                mutate(data)
            return httpx.Response(200, json=data)
        return httpx.Response(200, json=load_fixture("web_ddg.json"))
    api.use(handler)


def test_image_descriptions_from_titles(api):
    image_api(api)
    titles = [i["title"] for i in load_fixture("images_ddg.json")["image"]]
    images = api.post("/search", json={"query": "python logo", "include_images": True,
                                       "include_image_description": True}).json()["images"]
    assert len(images) == 10 and all(set(i) == {"url", "description"} for i in images)
    assert images[0]["description"] == titles[0] and all(i["description"] in titles for i in images)


def test_image_without_title_gets_empty_description(api):
    def drop_titles(data):
        for img in data["image"]:
            img["title"] = None
    image_api(api, drop_titles)
    r = api.post("/search", json={"query": "python logo", "include_images": True, "include_image_description": True})
    assert r.status_code == 200 and all(i["description"] == "" for i in r.json()["images"])


def test_image_description_without_include_images(api):
    image_api(api)
    images = api.post("/search", json={"query": "python logo", "include_image_description": True}).json()["images"]
    assert images == []



# --- vision image descriptions ------------------------------------------------------------

def vision(api, monkeypatch, describe):
    """Turn on the vision model and replace describe_image with describe(img) -> str | None."""
    import llm.images as images

    async def fake(llm, client, img, model, timeout):
        return await describe(img)

    monkeypatch.setattr(images, "describe_image", fake)
    api.app.state.llm_vision_model = "vision-1"
    api.llm(FakeLLM())
    image_api(api)


IMG_BODY = {"query": "python logo", "include_images": True, "include_image_description": True}


def test_vision_descriptions_used(api, monkeypatch):
    async def describe(img):
        return f"seen {img.thumbnail[-12:]}"
    vision(api, monkeypatch, describe)
    images = api.post("/search", json=IMG_BODY).json()["images"]
    assert len(images) == 10 and all(i["description"].startswith("seen ") for i in images)


def test_failed_description_falls_back_to_title(api, monkeypatch):
    titles = [i["title"] for i in load_fixture("images_ddg.json")["image"]]
    calls = []

    async def describe(img):
        calls.append(img)
        return None if len(calls) % 2 else "from the model"
    vision(api, monkeypatch, describe)
    images = api.post("/search", json=IMG_BODY).json()["images"]
    assert {i["description"] for i in images} - {"from the model"} <= set(titles)
    assert sum(i["description"] == "from the model" for i in images) == 5


def test_images_past_the_limit_get_titles(api, monkeypatch):
    async def describe(img):
        return "from the model"
    vision(api, monkeypatch, describe)
    api.app.state.max_img_desc = 3
    images = api.post("/search", json=IMG_BODY).json()["images"]
    assert [i["description"] == "from the model" for i in images] == [True] * 3 + [False] * 7


def test_no_vision_model_gives_titles(api, monkeypatch):
    async def describe(img):
        raise AssertionError("must not be called")
    vision(api, monkeypatch, describe)
    api.app.state.llm_vision_model = ""
    titles = [i["title"] for i in load_fixture("images_ddg.json")["image"]]
    images = api.post("/search", json=IMG_BODY).json()["images"]
    assert len(images) == 10 and all(i["description"] in titles for i in images)


def test_vision_off_without_include_image_description(api, monkeypatch):
    async def describe(img):
        raise AssertionError("must not be called")
    vision(api, monkeypatch, describe)
    images = api.post("/search", json={"query": "python logo", "include_images": True}).json()["images"]
    assert len(images) == 10 and all(isinstance(i, str) for i in images)


def test_descriptions_run_with_the_answer(api, monkeypatch):
    async def describe(img):
        await asyncio.sleep(0.4)
        return "d"
    vision(api, monkeypatch, describe)
    api.llm(FakeLLM(delay=0.4))
    t = time.perf_counter()
    body = api.post("/search", json={**IMG_BODY, "include_answer": True}).json()
    assert body["answer"] == "LLM answer." and body["images"][0]["description"] == "d"
    assert time.perf_counter() - t < 0.75

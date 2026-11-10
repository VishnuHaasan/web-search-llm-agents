import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import time
import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from starlette.requests import Request

from fetch.extract import Extractor
from fetch.safe_fetch import make_client
from llm.answer import generate_answer, generate_follow_ups
from llm.client import LLMClient
from llm.images import describe_images
from models.schema import SearchResponse, SearchQuery, ResultSchema, ImageSchema
from rank.rank import Ranker
from search.filters import Filters
from search.fourget.provider import FourGetProvider
from search.auto import apply_auto_parameters
from search.merge import cap_per_domain, CAP_PER_DOMAIN
from search.orchestrator import run_search
from search.enrich import apply_extraction
from models.schema import ExtractResponse, ExtractQuery, ExtractResult, FailedResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    root_path = Path(__file__).resolve().parents[2]
    load_dotenv(root_path.joinpath(".env"))
    fourget_url = os.getenv("FOURGET_URL", "http://localhost:80/api/v1")
    provider_timeout = float(os.getenv("PROVIDER_TIMEOUT", "8"))
    httpx_trust_env = int(os.getenv("HTTPX_TRUST_ENV", "0"))
    connect_timeout = float(os.getenv("FETCH_CONNECT_TIMEOUT", "3"))
    read_timeout = float(os.getenv("FETCH_READ_TIMEOUT", "5"))
    page_timeout = float(os.getenv("PAGE_TIMEOUT", "8"))
    min_text = int(os.getenv("MIN_TEXT", "200"))
    max_concurrent_page_fetches = int(os.getenv("MAX_CONCURRENT_PAGE_FETCHES", "8"))
    max_request_budget = float(os.getenv("MAX_REQUEST_BUDGET", "15"))
    min_extract_time = float(os.getenv("MIN_EXTRACT_TIME", "2"))
    max_page_chars = int(os.getenv("MAX_PAGE_CHARS", "100000"))
    max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    bm25_k1 = float(os.getenv("BM25_K1", "1.5"))
    bm25_b = float(os.getenv("BM25_B", "0.75"))
    ranker_alpha = float(os.getenv("RANKER_ALPHA", "0.5"))
    rerank_model = os.getenv("RERANK_MODEL", "")
    rerank_per_page = int(os.getenv("RERANK_PER_PAGE", "5"))
    model_cache_dir = root_path / os.getenv("MODEL_CACHE_DIR", "data/models")
    data_dir = root_path / "data"
    llm_url = os.getenv("LLM_URL", "")
    llm_model = os.getenv("LLM_MODEL", "")
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_timeout = float(os.getenv("LLM_TIMEOUT", "20"))
    llm_trust_env = int(os.getenv("LLM_TRUST_ENV", "0"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    app.state.llm_vision_model = os.getenv("LLM_VISION_MODEL", "")
    app.state.max_img_desc = int(os.getenv("MAX_IMAGE_DESCRIPTIONS", "10"))
    app.state.img_desc_timeout = float(os.getenv("IMAGE_DESCRIPTION_TIMEOUT", "8"))
    Filters.load(data_dir=data_dir)
    client = httpx.AsyncClient(base_url=fourget_url, timeout=provider_timeout, trust_env=bool(httpx_trust_env))
    fetch_client = make_client(read_timeout, connect_timeout)
    app.state.providers = [FourGetProvider(client=client)]
    app.state.extractor = Extractor(client=fetch_client, page_timeout=page_timeout, min_text_filter=min_text,
                                    concurrency=max_concurrent_page_fetches, max_page_chars=max_page_chars)
    app.state.max_request_budget = max_request_budget
    app.state.min_extract_time = min_extract_time
    app.state.ranker = Ranker(k1=bm25_k1, b=bm25_b, max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
                              alpha=ranker_alpha, model_name=rerank_model, model_cache_dir=str(model_cache_dir),
                              rerank_per_page=rerank_per_page)
    llm_client = None
    app.state.llm_client = None
    app.state.fetch_client = fetch_client
    if llm_url and llm_model:
        llm_client = httpx.AsyncClient(
            base_url=llm_url,
            headers={"Authorization": f"Bearer {llm_api_key}"},
            timeout=llm_timeout,
            trust_env=bool(llm_trust_env),

        )
        app.state.llm_client = LLMClient(
            client=llm_client,
            model=llm_model,
            timeout=llm_timeout,
            temperature=llm_temperature,
        )
    yield
    await client.aclose()
    await fetch_client.aclose()
    if llm_client:
        await llm_client.aclose()

app = FastAPI(lifespan=lifespan)


def raw_content(c, mode) -> str | None:
    """include_raw_content: False -> none, "text" -> plain, True / "markdown" -> markdown (plain if none)."""
    if not mode:
        return None
    if mode == "text":
        return c.text
    return c.markdown or c.text


@app.post("/search", response_model=SearchResponse)
async def search(q: SearchQuery, request: Request) -> SearchResponse:
    start = time.perf_counter()
    auto = None
    if q.auto_parameters:
        q, auto = apply_auto_parameters(q)
    merged = await run_search(q, request.app.state.providers)
    if not merged.candidates and merged.errors:
        raise HTTPException(502, detail=[e.model_dump() for e in merged.errors])
    remaining = max(request.app.state.max_request_budget - (time.perf_counter() - start), request.app.state.min_extract_time)
    want_md = q.include_raw_content in (True, "markdown")
    extracted = await request.app.state.extractor.extract_many([c.url for c in merged.candidates], remaining,
                                                               markdown=want_md)
    candidates = apply_extraction(merged.candidates, extracted, q)
    candidates = await asyncio.to_thread(request.app.state.ranker.rank_candidates, candidates=candidates, q=q)
    candidates = cap_per_domain(candidates, CAP_PER_DOMAIN)
    for prev, cur in zip(candidates, candidates[1:]):
        cur.score = min(cur.score, prev.score)
    top_candidates = candidates[: q.max_results]
    llm = request.app.state.llm_client

    async def _answer() -> str | None:
        if not q.include_answer or not llm:
            return None
        return await generate_answer(llm, q, top_candidates)

    async def _follow_ups() -> list[str]:
        if not q.include_follow_up_questions or not llm:
            return []
        return await generate_follow_ups(llm, q, top_candidates)

    async def _descriptions() -> list[str | None]:
        if not (q.include_images and q.include_image_description and llm and request.app.state.llm_vision_model and merged.images):
            return [None] * len(merged.images)
        return await describe_images(llm, request.app.state.fetch_client, merged.images, request.app.state.llm_vision_model, request.app.state.max_img_desc, request.app.state.img_desc_timeout)

    answer, follow_ups, descriptions = await asyncio.gather(_answer(), _follow_ups(), _descriptions())    # answer, followups and description llm calls run together
    if answer is None and q.include_answer and merged.instant_answers:
        answer = merged.instant_answers[0].text

    if q.include_image_description:
        images = [ImageSchema(url=i.url, description=d or i.title or "")
            for i, d in zip(merged.images, descriptions, strict=True)]
    else:
        images = [i.url for i in merged.images]
    return SearchResponse(
        query=q.query,
        results=[ResultSchema(
            title=c.title,
            url=c.url,
            content=c.content or "",
            score=round(c.score, 4),
            published_date=c.published_at.strftime("%Y-%m-%d") if c.published_at else None,
            raw_content=raw_content(c, q.include_raw_content),
            favicon=c.favicon if q.include_favicon else None,

        ) for c in top_candidates],
        images=images,
        response_time=round(time.perf_counter() - start, 2),
        request_id=str(uuid.uuid4()),
        answer=answer,
        follow_up_questions=follow_ups,
        auto_parameters=auto,
    )

@app.post("/extract", response_model=ExtractResponse)
async def extract(q: ExtractQuery, request: Request) -> ExtractResponse:
    start = time.perf_counter()
    urls = list(dict.fromkeys(q.urls))
    timeout = q.timeout or (30.0 if q.extract_depth == "advanced" else 10.0)
    want_md = q.format == "markdown"
    extracted = await request.app.state.extractor.extract_many(urls, timeout, markdown=want_md)
    return ExtractResponse(
        results=[ExtractResult(url=e.url, raw_content=(e.markdown if want_md else None) or e.text,
                               favicon=e.favicon if q.include_favicon else None)
                 for e in extracted if e.text],
        failed_results=[FailedResult(url=e.url, error=e.error) for e in extracted if not e.text],
        response_time=round(time.perf_counter() - start, 2),
        request_id=str(uuid.uuid4()),
    )


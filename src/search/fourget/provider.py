import asyncio
from datetime import timedelta, datetime, timezone

import httpx
from models.schema import SearchQuery
from search.filters import Filters
from search.fourget.scraper import FourGetScraperPool
from search.operators import operator_queries
from search.query import SearchProvider
from search.schema import FourGetResponse, FourGetRequest, Candidate, ProviderHit, InstantAnswer, ImageCandidate
from utils.errors import FourGetError


class FourGetProvider(SearchProvider[FourGetResponse]):
    name = "4get"
    _MAX_IMAGES = 10
    _MAX_RETRIES = 2

    def __init__(self, client: httpx.AsyncClient, max_pages: int = 3):
        super().__init__(max_pages)
        self.client = client
        self.scraper_pool = FourGetScraperPool()

    def build_queries(self, q: SearchQuery) -> list[str]:
        return operator_queries(q)

    async def _get(self, endpoint: str, params: dict, scraper: str) -> tuple[FourGetResponse, str | None]:
        try:
            resp = await self.client.get(endpoint, params=params)
            resp.raise_for_status()
            page = FourGetResponse.model_validate(resp.json())
            page.scraper = scraper
        except httpx.ConnectError as e:
            raise FourGetError(f"4get unreachable : {e!r}", FourGetError.DOWN) from e
        except httpx.TimeoutException as e:
            raise FourGetError(f"4get timeout : {e!r}", FourGetError.TRANSIENT) from e
        except httpx.TransportError as e:
            raise FourGetError(f"4get network error: {e!r}", FourGetError.TRANSIENT) from e
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            kind = FourGetError.TRANSIENT if code >= 500 else FourGetError.BLOCKED if code == 429 else FourGetError.INVALID
            raise FourGetError(f"HTTP : {code}", kind) from e
        except ValueError as e:
            raise FourGetError(f"bad response : {e!r}", FourGetError.BLOCKED) from e
        if page.status != "ok":
            raise FourGetError(page.status, FourGetError.kind_from_message(page.status))
        cursor = f"{page.scraper}|{page.npt}" if page.npt else None
        return page, cursor

    def _handle_web_exc(self, scraper : str, exc : Exception, no_retry: bool = True):
        if isinstance(exc, FourGetError):
            if exc.kind in (exc.INVALID, exc.DOWN):
                raise exc
            else:
                self.scraper_pool.report_failure(scraper, exc.kind)
                if no_retry:
                    raise exc
        else:
            raise exc

    @classmethod
    def _construct_params(cls, q: SearchQuery, query: str, scraper: str) -> tuple[FourGetRequest, dict]:
        start, end = Filters.date_window(q)
        req = FourGetRequest(
            s=query,
            scraper=scraper,
            country=q.country,
            nsfw="no" if q.safe_search else None,
            newer=start.date().isoformat() if start else None,
            older=(end - timedelta(days=1)).date().isoformat() if end else None,
        )
        params = req.model_dump(exclude_none=True)
        return req, params

    def _construct_img_params(self, req: FourGetRequest) -> dict:
        p = {
            "s": req.s,
            "scraper": req.scraper,
        }
        if req.nsfw is not None:
            p["nsfw"] = req.nsfw
        return p

    async def fetch_page(self, q: SearchQuery, query: str, cursor: str | None) -> tuple[FourGetResponse, str | None]:
        endpoint = "/news" if q.topic == "news" else "/web"
        page = None
        if cursor is None:
            for i_try in range(self._MAX_RETRIES):
                order = self.scraper_pool.order(q)
                scraper = self.scraper_pool.pick(order)
                req, params = self._construct_params(q, query, scraper)
                if q.include_images:
                    img_params = self._construct_img_params(req)
                    web_resp, img_resp = await asyncio.gather(self._get(endpoint, params, scraper),
                                        self.client.get("/images", params=img_params),
                                                              return_exceptions=True)
                    if isinstance(web_resp, Exception):
                        self._handle_web_exc(scraper, web_resp, no_retry=bool(i_try))
                        continue
                    page, cursor = web_resp
                    if not isinstance(img_resp, Exception):
                        try:
                            img_resp.raise_for_status()
                            images = FourGetResponse.model_validate(img_resp.json())
                            if images.status == "ok":
                                page.image.extend(images.image)
                        except Exception as e:
                            pass
                    self.scraper_pool.report_success(scraper)
                    break
                else:
                    try:
                        page, cursor = await self._get(endpoint, params, scraper)
                        self.scraper_pool.report_success(scraper)
                        break
                    except FourGetError as exc:
                        self._handle_web_exc(scraper, exc, no_retry=bool(i_try))

        else:
            scraper, npt = cursor.split("|", 1)
            params = {"npt": npt}
            try:
                page, cursor = await self._get(endpoint, params, scraper)
                self.scraper_pool.report_success(scraper)
            except FourGetError as exc:
                self._handle_web_exc(scraper, exc, no_retry=True)
        assert page is not None
        return page, cursor

    def to_candidates(self, q: SearchQuery, page: FourGetResponse, rank_offset: int) -> list[Candidate]:
        results = [(r, "web") for r in page.web] + [(r, "news") for r in page.news]
        candidates = []
        for i, (result, kind) in enumerate(results):
            rank = rank_offset + i + 1
            if not Filters.is_fetchable(result.url):
                continue

            try:
                can_url = Filters.canonicalize_url(result.url)
            except ValueError:
                continue

            extra = {}
            if kind == "web":
                if result.sublink:
                    extra["sublink"] = result.sublink
                if result.table:
                    extra["table"] = result.table
            else:
                if result.author:
                    extra["author"] = result.author

            candidate = Candidate(
                url=result.url,
                canonical_url=can_url,
                title=result.title,
                snippet=result.description if result.description else None,
                published_at=datetime.fromtimestamp(result.date, timezone.utc) if result.date else None,
                kind=kind,
                thumbnail=result.thumb.url if kind == "news" and result.thumb else None,
                hits=[ProviderHit(
                    provider=self.name,
                    engine=page.scraper,
                    rank=rank,
                )],
                extra=extra,
            )
            candidates.append(candidate)

        return candidates

    def to_instant_answers(self, page: FourGetResponse) -> list[InstantAnswer]:
        answers = []
        for answer in page.answer:
            text = " ".join(desc.value for desc in answer.description)
            if not text.strip():
                continue
            ans = InstantAnswer(
                provider=self.name,
                title=answer.title,
                text=text,
                url=answer.url,
            )
            answers.append(ans)
        return answers

    def spelling_of(self, page: FourGetResponse) -> str | None:
        if not page.spelling or page.spelling.type == "no_correction":
            return None
        def _remove_substr(s: str, sub: str) -> str:
            idx = s.find(sub)
            if idx != -1:
                return s[:idx]
            return s
        _br = "(site:"
        _neg = "-site:"
        _plain = "site:"
        using = page.spelling.using
        if not using:
            return None
        out = _remove_substr(_remove_substr(_remove_substr(using, _br), _neg), _plain).strip()
        return out if out else None

    def to_images(self, page: FourGetResponse) -> list[ImageCandidate]:
        candidates = []
        for img in page.image:
            if len(candidates) >= self._MAX_IMAGES:
                break
            if not img.source:
                continue
            best_source = max(img.source, key=lambda s: (s.height or 0) * (s.width or 0))
            thumbnail = min(img.source, key=lambda s: (s.height or 0) * (s.width or 0))
            candidates.append(ImageCandidate(
                url=best_source.url,
                title=img.title,
                source_page=img.url,
                provider=self.name,
                thumbnail=thumbnail.url,
            ))
        return candidates






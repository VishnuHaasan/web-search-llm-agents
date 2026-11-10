from abc import ABC, abstractmethod

import math
import time

from models.schema import SearchQuery
from search.schema import SearchBatch, Candidate, ImageCandidate, InstantAnswer, ProviderError


class SearchProvider[RawResponse](ABC):

    _MULTIPLIER_MAP : dict[str, float] = {
        "ultra-fast": 1.2,
        "fast": 2,
        "basic": 2,
        "advanced": 3
    }

    _MAX_ITEMS_TO_FETCH = 60

    _TIME_BUDGET = 10

    name: str
    def __init__(self, max_pages: int = 3):
        self.max_pages = max_pages

    @abstractmethod
    async def fetch_page(self, q: SearchQuery, query: str, cursor: str | None) -> tuple[RawResponse, str | None]:
        raise NotImplementedError

    @abstractmethod
    def to_candidates(self, q: SearchQuery, page: RawResponse, rank_offset: int) -> list[Candidate]:
        raise NotImplementedError

    def to_images(self, page: RawResponse) -> list[ImageCandidate]:
        return []

    def to_instant_answers(self, page: RawResponse) -> list[InstantAnswer]:
        return []

    def spelling_of(self, page: RawResponse) -> str | None:
        return None

    async def search(self, q: SearchQuery) -> SearchBatch:
        start = time.perf_counter()
        queries = self.build_queries(q)
        if not queries:
            return SearchBatch(
                query=q.query,
                effective_query=[]
            )
        max_candidates = self.candidate_target(q)
        budget_per_query = math.ceil(max_candidates / len(queries))
        all_candidates = []
        images : list[ImageCandidate] = []
        answers : list[InstantAnswer] = []
        spelling : str | None = None
        errors : list[ProviderError] = []

        seen = set()
        for idx, query in enumerate(queries):
            cursor = None
            rank = 0
            collected_candidates = 0
            n_page = 0
            while collected_candidates < budget_per_query and n_page < self.max_pages:
                if time.perf_counter() - start > self._TIME_BUDGET:
                    break
                try:
                    page, cursor = await self.fetch_page(q, query, cursor)
                except Exception as e:
                    errors.append(ProviderError(provider=self.name, message=f"{query!r}: {e}"))
                    break
                candidates = self.to_candidates(q, page, rank)
                prev_candidates = collected_candidates
                for candidate in candidates:
                    if candidate.canonical_url not in seen:
                        seen.add(candidate.canonical_url)
                        all_candidates.append(candidate)
                        collected_candidates += 1
                    rank += 1

                if n_page == 0:
                    if q.include_images:
                        images.extend(self.to_images(page))
                    if idx == 0:
                        if q.include_answer:
                            answers.extend(self.to_instant_answers(page))
                        spelling = self.spelling_of(page)
                n_page += 1

                if cursor is None or collected_candidates == prev_candidates:
                    break

        took = (time.perf_counter() - start) * 1000
        return SearchBatch(
            query=q.query,
            effective_query=queries,
            candidates=all_candidates,
            images=images,
            instant_answers=answers,
            spelling_correction=spelling,
            errors=errors,
            elapsed_ms=took,
        )

    @abstractmethod
    def build_queries(self, q: SearchQuery) -> list[str]:
        raise NotImplementedError

    @classmethod
    def candidate_target(cls, q: SearchQuery) -> int:
        multiplier = cls._MULTIPLIER_MAP[q.search_depth]
        if any((q.include_domains, q.exclude_domains, q.time_range, q.start_date, q.end_date, q.exact_match)):
            multiplier += 0.5

        return min(math.ceil(multiplier * q.max_results), cls._MAX_ITEMS_TO_FETCH)





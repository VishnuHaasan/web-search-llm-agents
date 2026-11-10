from collections import defaultdict

from models.schema import SearchQuery
from search.filters import Filters
from search.query import SearchProvider
from search.schema import SearchBatch, Candidate, ImageCandidate

_RRF_K = 60
CAP_PER_DOMAIN = 3

def _group_by_canonical_url(batches: list[SearchBatch]):
    grouped_batches : dict[str, list[Candidate]] = {}
    for batch in batches:
        for candidate in batch.candidates:
            grouped_batches.setdefault(candidate.canonical_url, []).append(candidate)
    return grouped_batches

def _merge_extra(extra: dict, new: dict) -> dict:
    for key, value in new.items():
        if key not in extra:
            extra[key] = value

    return extra

def _merge_group_to_candidate(grouped_batches: dict[str, list[Candidate]]) -> list[Candidate]:
    all_candidates: list[Candidate] = []
    for canonical, candidates in grouped_batches.items():
        first_candidate = candidates[0]
        hits = []
        https = None
        count_dict = defaultdict(int)
        url_mx = 1
        url = first_candidate.url
        rank_mn = first_candidate.hits[0].rank
        title = first_candidate.title
        snip_mx = len(first_candidate.snippet) if first_candidate.snippet else 0
        snip = first_candidate.snippet
        published_at = None
        kind = "web"
        thumbnail = None
        extra = {}
        for candidate in candidates:
            hits = hits + candidate.hits
            if not https and candidate.url.startswith("https://"):
                https = candidate.url

            count_dict[candidate.url] += 1
            if count_dict[candidate.url] > url_mx:
                url_mx = count_dict[candidate.url]
                url = candidate.url

            for hit in candidate.hits:
                if hit.rank < rank_mn:
                    rank_mn = hit.rank
                    title = candidate.title

            if candidate.snippet and len(candidate.snippet) > snip_mx:
                snip_mx = len(candidate.snippet)
                snip = candidate.snippet

            if candidate.published_at and (published_at is None or candidate.published_at < published_at):
                published_at = candidate.published_at

            if candidate.thumbnail and not thumbnail:
                thumbnail = candidate.thumbnail

            if candidate.kind == "news":
                kind = "news"

            extra = _merge_extra(extra, candidate.extra)

        new_candidate = Candidate(
            url=https if https else url,
            canonical_url=canonical,
            title=title,
            snippet=snip if snip and len(snip) > 3 else None,
            published_at=published_at,
            kind=kind,
            hits=hits,
            thumbnail=thumbnail,
            extra=extra,
        )
        all_candidates.append(new_candidate)

    return all_candidates

def _set_rrf_score(candidates: list[Candidate]):

    for candidate in candidates:
        candidate.prior_score = sum(1 / (_RRF_K + h.rank) for h in candidate.hits)

def _filter_candidates(candidates: list[Candidate], q: SearchQuery) -> list[Candidate]:
    start, end = Filters.date_window(q)
    include, exclude = set(q.include_domains), set(q.exclude_domains)
    return [candidate for candidate in candidates if Filters.passes_filters(candidate, q, start, end, include, exclude)]

def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda candidate: (-candidate.prior_score, min([hit.rank for hit in candidate.hits])))

def _deduplicate_images_by_url_and_limit(images: list[ImageCandidate]) -> list[ImageCandidate]:
    deduplicated_images = []
    url_set = set()
    for image in images:
        if image.url not in url_set:
            deduplicated_images.append(image)
            url_set.add(image.url)

    return deduplicated_images[:10]

def _dedup_eff_queries(eff_queries: list[str]) -> list[str]:
    deduplicated_queries = []
    for eff in eff_queries:
        if eff not in deduplicated_queries:
            deduplicated_queries.append(eff)

    return deduplicated_queries

def cap_per_domain(candidates: list[Candidate], cap: int) -> list[Candidate]:
    counts_per_domain = {}
    top, bottom = [], []
    for candidate in candidates:
        domain = Filters.domain_of(candidate.url)
        counts_per_domain[domain] = counts_per_domain.get(domain, 0) + 1
        if counts_per_domain[domain] <= cap:
            top.append(candidate)
        else:
            bottom.append(candidate)

    return top + bottom

def merge_batches(batches: list[SearchBatch], q: SearchQuery) -> SearchBatch:

    grouped_batches = _group_by_canonical_url(batches)
    grouped_candidates = _merge_group_to_candidate(grouped_batches)
    _set_rrf_score(grouped_candidates)
    filtered_candidates = _filter_candidates(grouped_candidates, q)
    sorted_candidates = _sort_candidates(filtered_candidates)
    capped = cap_per_domain(sorted_candidates, CAP_PER_DOMAIN)
    limit = SearchProvider.candidate_target(q)
    images = _deduplicate_images_by_url_and_limit([image for batch in batches for image in batch.images])
    eff_queries = _dedup_eff_queries([eff_query for batch in batches for eff_query in batch.effective_query])

    out = SearchBatch(
        query=q.query,
        effective_query=eff_queries,
        candidates=capped[:limit],
        images=images,
        instant_answers=[answer for batch in batches for answer in batch.instant_answers],
        spelling_correction=next((batch.spelling_correction for batch in batches if batch.spelling_correction), None),
        errors=[error for batch in batches for error in batch.errors],
        elapsed_ms=max([batch.elapsed_ms for batch in batches], default=0.0)
    )
    return out



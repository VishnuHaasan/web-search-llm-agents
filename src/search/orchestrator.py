import asyncio

from models.schema import SearchQuery
from search.merge import merge_batches
from search.query import SearchProvider
from search.schema import SearchBatch, ProviderError


async def run_search(q: SearchQuery, providers: list[SearchProvider]) -> SearchBatch:

    promises = await asyncio.gather(*[provider.search(q) for provider in providers], return_exceptions=True)
    batches, errors = [], []
    for provider, results in zip(providers, promises):
        if isinstance(results, Exception):
            error = ProviderError(provider=provider.name, message=repr(results))
            errors.append(error)
        else:
            batches.append(results)

    merged_batches = merge_batches(batches, q)
    merged_batches.errors.extend(errors)
    return merged_batches

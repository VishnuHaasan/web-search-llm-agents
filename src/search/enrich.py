import re
from fetch.extract import Extracted
from models.schema import SearchQuery
from search.filters import Filters
from search.schema import Candidate

def _quoted_phrases(query: str) -> list[str]:
    return [_normalize(phrase) for phrase in re.findall(r'"([^"]+)"', query)]

def _normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.casefold()).strip()

def _has_phrases(c: Candidate, phrases: list[str]) -> bool:
    text_to_check = c.title + " " + (c.text or "")
    normalized_text_to_check = _normalize(text_to_check)
    return all((f" {phrase} " in f" {normalized_text_to_check} " for phrase in phrases if phrase))

def apply_extraction(candidates: list[Candidate], extracted: list[Extracted], q: SearchQuery) -> list[Candidate]:

    start, end = Filters.date_window(q)
    filtered = []
    phrases = _quoted_phrases(q.query) if q.exact_match else []
    for c, e in zip(candidates, extracted, strict=True):
        c.favicon = e.favicon
        if e.text:
            c.text = e.text
            c.markdown = e.markdown
        else:
            c.text = c.snippet
            c.extract_error = e.error

        if c.published_at is None and e.published_at:
            c.published_at = e.published_at
            if not Filters.in_window(c, q, start, end):
                continue

        if not c.title and e.title:
            c.title = e.title

        if phrases and not _has_phrases(c, phrases):
            continue

        filtered.append(c)

    return filtered


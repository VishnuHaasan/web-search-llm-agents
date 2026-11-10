import re

from models.schema import SearchQuery
from rank.bm25 import BM25

_QUOTED = re.compile(r'"[^"]*"')
_NEWS = re.compile(r"\b(?:news|headlines?|breaking|announced|announcement|press release)\b")
_TIME_RANGES = [
    ("day", re.compile(r"\b(?:today|tonight|right now|breaking)\b")),
    ("week", re.compile(r"\b(?:this|past|last) week\b")),
    ("month", re.compile(r"\b(?:this|past|last) month\b")),
    ("year", re.compile(r"\b(?:this|past|last) year\b")),
]
_QUESTION_START = re.compile(r"^(?:how|why|what|which)\b")
_COMPARE = re.compile(r"\b(?:vs|versus|compare|difference)\b")


def _clean(query: str) -> str:
    """Lowercase query without "quoted phrases" and site: operators, so they never trigger a rule."""
    return BM25.SITE.sub(" ", _QUOTED.sub(" ", query)).casefold().strip()


def _time_range(text: str) -> str | None:
    for name, pattern in _TIME_RANGES:
        if pattern.search(text):
            return name
    return None


def _is_question(query: str, text: str) -> bool:
    return query.rstrip().endswith("?") or bool(_QUESTION_START.search(text) or _COMPARE.search(text))


def apply_auto_parameters(q: SearchQuery) -> tuple[SearchQuery, dict]:
    """Guess topic / time_range / search_depth from the query words; fields the client sent always win.

    Returns the updated query and the final values of the three fields, echoed as `auto_parameters`.
    """
    text = _clean(q.query)
    sent = q.model_fields_set
    update = {}

    if "topic" not in sent and _NEWS.search(text):
        update["topic"] = "news"
    if not ({"time_range", "start_date", "end_date"} & sent):     # the validator forbids time_range with dates
        time_range = _time_range(text)
        if time_range:
            update["time_range"] = time_range
    if "search_depth" not in sent and _is_question(q.query, text):
        update["search_depth"] = "advanced"

    q = q.model_copy(update=update)
    return q, {"topic": q.topic, "time_range": q.time_range, "search_depth": q.search_depth}

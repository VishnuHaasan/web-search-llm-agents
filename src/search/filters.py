from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit

import courlan
import tldextract

from models.schema import SearchQuery
from pathlib import Path
import re

from search.schema import Candidate


class Filters:

    _RANGE = {"day": 1, "week": 7, "month": 30, "year": 365}
    _DOMAIN_MATCH_RULE = re.compile(r"\*://(?:\*\.)?([a-z0-9.-]+)/\*")
    _EXTRACT = tldextract.TLDExtract(include_psl_private_domains=True)

    _blocklist: set[str] | None = None
    _no_extract: set[str] | None = None
    _allow_list: set[str] | None = None

    @classmethod
    def load(cls, data_dir: Path) -> None:
        cls._blocklist = cls._load_blocklist(data_dir / "blocklists")
        cls._no_extract = cls._load_plain(data_dir / "no-extract.txt")
        cls._allow_list = cls._load_plain(data_dir / "allow-list.txt")

    @classmethod
    def _load_blocklist(cls, directory: Path) -> set[str]:
        domains: set[str] = set()
        for f in directory.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                m = cls._DOMAIN_MATCH_RULE.fullmatch(line.strip().lower())
                if m:
                    domains.add(m.group(1).removeprefix("www."))
        return domains

    @staticmethod
    def _load_plain(path: Path) -> set[str]:
        lines = (l.strip().lower() for l in path.read_text(encoding="utf-8").splitlines())
        return {l.removeprefix("www.") for l in lines if l and not l.startswith("#")}

    @staticmethod
    def canonicalize_url(url: str) -> str:
        return (courlan.normalize_url(url, strict=True)
                .removeprefix("https://").removeprefix("http://")
                .removeprefix("www.").rstrip("/"))

    @classmethod
    def is_fetchable(cls, url: str) -> bool:
        if cls._blocklist is None or cls._no_extract is None:
            raise RuntimeError("Filters Load was not called")
        return courlan.check_url(url) is not None and (cls.matches_domains(url, cls._allow_list) or not cls.matches_domains(url, cls._no_extract)) and bool(cls._EXTRACT(url).suffix)

    @classmethod
    def domain_of(cls, url: str) -> str:
        return cls._EXTRACT(url).top_domain_under_public_suffix

    @classmethod
    def date_window(cls, q: SearchQuery) -> tuple[datetime | None, datetime | None]:
        if q.time_range:
            now = datetime.now(timezone.utc)
            return now - timedelta(days=cls._RANGE[q.time_range]), None
        start = datetime.combine(q.start_date, datetime.min.time(), timezone.utc) if q.start_date else None
        end = datetime.combine(q.end_date + timedelta(days=1), datetime.min.time(), timezone.utc) if q.end_date else None
        return start, end

    @staticmethod
    def matches_domains(url: str, domains: set[str]) -> bool:
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        labels = host.split(".")
        return any(".".join(labels[i:]) in domains for i in range(len(labels)))

    @classmethod
    def in_window(cls, c: Candidate, q: SearchQuery, start, end) -> bool:
        if start is None and end is None:
            return True
        if c.published_at is None:
            return q.topic != "news"
        t = c.published_at if c.published_at.tzinfo else c.published_at.replace(tzinfo=timezone.utc)
        return (start is None or t >= start) and (end is None or t < end)

    @classmethod
    def passes_filters(cls, candidate : Candidate, q: SearchQuery, date_start, date_end, include_set, exclude_set) -> bool:
        return (cls.is_fetchable(candidate.url)
                and (cls.matches_domains(candidate.url, cls._allow_list) or not cls.matches_domains(candidate.url, cls._blocklist))
                and not cls.matches_domains(candidate.url, exclude_set)
                and (not include_set or cls.matches_domains(candidate.url, include_set))
                and cls.in_window(candidate, q, date_start, date_end))


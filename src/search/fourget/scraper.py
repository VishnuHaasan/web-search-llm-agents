import logging
import time
from dataclasses import dataclass
from typing import Sequence

from models.schema import SearchQuery
from utils.errors import FourGetError


@dataclass
class ScraperState:
    name: str
    fail_count: int = 0
    cooldown_until: float = 0.0

class FourGetScraperPool:
    _WEB = ["ddg", "startpage", "qwant", "yandex", "marginalia", "brave"]
    _NEWS = ["ddg"]

    def __init__(self):
        self.scraper_states = {
            scraper: ScraperState(
                name=scraper,
            ) for scraper in set(FourGetScraperPool._WEB + FourGetScraperPool._NEWS)
        }

    def report_failure(self, scraper: str, kind: str) -> None:
        st = self.scraper_states[scraper]
        st.fail_count += 1
        base, cap = (300, 3600) if kind == "blocked" else (20, 300)
        st.cooldown_until = time.monotonic() + min(base * 2 ** (st.fail_count - 1), cap)

    def report_success(self, scraper: str) -> None:
        st = self.scraper_states[scraper]
        st.fail_count = 0
        st.cooldown_until = 0

    def order(self, q: SearchQuery) -> list[str]:
        if q.engines:
            available_engines = []
            all_engines = self._NEWS if q.topic == "news" else self._WEB
            for engine in q.engines:
                if engine in all_engines:
                    available_engines.append(engine)
            if not available_engines:
                raise FourGetError("all engines are invalid", kind="invalid")
            missing = list(set(q.engines) - set(available_engines))
            if missing:
                logging.warning(f"Not all engines used, these engines are invalid : {missing}")
            return available_engines
        return self._NEWS if q.topic == "news" else self._WEB

    def pick(self, order : Sequence[str]) -> str:
        now = time.monotonic()
        for s in order:
            if self.scraper_states[s].cooldown_until <= now:
                return s
        raise FourGetError("all scrapers are in cooldown", kind="blocked")









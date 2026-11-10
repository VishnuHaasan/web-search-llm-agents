from datetime import datetime
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, field_validator, Field

class Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")

class Spelling(Lenient):
    type: Literal["no_correction", "including", "not_many"]
    using: Optional[str] = None
    correction: Optional[str] = None

class Description(Lenient):
    type: str
    value: str
    url: Optional[str] = None

class Thumbnail(Lenient):
    ratio: Optional[str] = None
    url: Optional[str] = None

def _table(v: Any) -> Dict[Any, Any]:
    return {} if v in (None, []) else v

class Answer(Lenient):
    title: str
    description: List[Description]
    url: Optional[str] = None
    thumb: Optional[str] = None
    table: Dict[str, str] = {}
    sublink: Dict[str, str] = {}
    _t = field_validator("table", "sublink", mode="before")(lambda cls, v: _table(v))

class WebResult(Lenient):
    title: str
    description: Optional[str] = None
    url: str
    date: Optional[int] = None
    type: str = "web"
    sublink: List[Dict[str, Any]] = []
    table: Dict[str, str] = {}
    _t = field_validator("table", mode="before")(lambda cls, v: _table(v))

class ImageSource(Lenient):
    url: str
    width: Optional[int] = None
    height: Optional[int] = None

class ImageResult(Lenient):
    title: Optional[str] = None
    source: List[ImageSource]
    url: str

class NewsResult(Lenient):
    title: str
    url: str
    description: Optional[str] = None
    author: Optional[str] = None
    date: Optional[int] = None          # unix seconds, same as WebResult.date
    thumb: Optional[Thumbnail] = None

class FourGetResponse(Lenient):
    status: str
    spelling: Optional[Spelling] = None
    npt: Optional[str] = None
    answer: List[Answer] = []
    web: List[WebResult] = []
    image: List[ImageResult] = []
    video: List[Dict[str, Any]] = []
    news: List[NewsResult] = []
    related: List[str] = []
    scraper: Optional[str] = None

class FourGetRequest(BaseModel):
    s: str
    npt: Optional[str] = None
    scraper: Optional[str] = None
    country: Optional[str] = None
    nsfw: Optional[Literal["yes", "maybe", "no"]] = None
    newer: Optional[str] = None
    older: Optional[str] = None


class ProviderHit(BaseModel):
    provider: str
    engine: Optional[str] = None
    rank: int

class Candidate(BaseModel):
    url: str
    canonical_url: str
    title: str
    snippet: Optional[str] = None
    published_at: Optional[datetime] = None
    kind: Literal["web", "news"] = "web"
    hits: List[ProviderHit] = []
    prior_score: float = 0.0
    thumbnail: Optional[str] = None
    extra: Dict[str, Any] = {}
    text: Optional[str] = None
    extract_error: Optional[str] = None
    content: Optional[str] = None
    score: float = 0.0
    markdown: Optional[str] = None
    favicon: Optional[str] = None

class ImageCandidate(BaseModel):
    url: str
    title: Optional[str] = None
    source_page: Optional[str] = None
    provider: str
    thumbnail: Optional[str] = None

class InstantAnswer(BaseModel):
    provider: str
    title: str
    text: str
    url: Optional[str] = None

class ProviderError(BaseModel):
    provider: str
    message: str

class SearchBatch(BaseModel):
    query: str
    effective_query: List[str]
    candidates: List[Candidate] = []
    images: List[ImageCandidate] = []
    instant_answers: List[InstantAnswer] = []
    spelling_correction: Optional[str] = None
    errors: List[ProviderError] = []
    elapsed_ms: float = 0.0

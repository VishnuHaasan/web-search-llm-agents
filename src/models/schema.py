import re
import datetime
from typing import Literal, List, Optional, Dict, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator, field_validator

_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")

class SearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    search_depth: Literal["ultra-fast", "fast", "basic", "advanced"] = "basic"
    max_results: int = Field(ge=1, le=20, default=5)
    chunks_per_source: int = Field(ge=1, le=3, default=3)
    topic: Literal["news", "finance", "general", "technology"] = "general"
    include_domains: List[str] = []
    exclude_domains: List[str] = []
    time_range: Optional[Literal["year", "month", "week", "day"]] = None
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    country: Optional[str] = None
    include_answer: bool | Literal["basic", "advanced"] = False
    include_raw_content: bool | Literal["markdown", "text"] = False
    include_images: bool = False
    include_image_description: bool = False
    include_favicon: bool = False
    exact_match: bool = False
    auto_parameters: bool = False
    safe_search: bool = False
    engines: List[str] = []
    include_follow_up_questions: bool = False

    @field_validator("include_domains", "exclude_domains", mode="after")
    @classmethod
    def domains_validator(cls, v):
        out: list[str] = []
        for d in v:
            d = d.strip()
            host = (urlsplit(d if "://" in d else "//" + d).hostname or "").removeprefix("www.")
            if not _HOST.fullmatch(host):
                raise ValueError(f"Invalid domain format: {d!r}")
            if host not in out:
                out.append(host)
        return out

    @model_validator(mode="after")
    def _check_dates(self):
        if self.time_range and (self.start_date or self.end_date):
            raise ValueError("use either time_range or start_date/end_date, not both")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self

class ImageSchema(BaseModel):
    url: str
    description: str

class ResultSchema(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None
    favicon: Optional[str] = None
    published_date: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    images: List[str] | List[ImageSchema] = []
    results: List[ResultSchema] = []
    response_time: float
    auto_parameters: Optional[Dict[str, Any]] = None
    request_id: str
    follow_up_questions: List[str] = []

class ExtractQuery(BaseModel):
    urls: List[str] = Field(min_length=1, max_length=20)
    extract_depth: Literal["basic", "advanced"] = "basic"
    format: Literal["markdown", "text"] = "markdown"
    timeout: Optional[float] = Field(default=None, ge=1, le=60)
    include_images: bool = False
    include_favicon: bool = False
    query: Optional[str] = None
    chunks_per_source: int = Field(default=3, ge=1, le=5)
    include_usage: bool = False

    @field_validator("urls", mode="before")
    @classmethod
    def _one_url(cls, v):
        return [v] if isinstance(v, str) else v       # "https://a.com" → ["https://a.com"]

class ExtractResult(BaseModel):
    url: str
    raw_content: str
    images: List[str] = []
    favicon: Optional[str] = None

class FailedResult(BaseModel):
    url: str
    error: str

class ExtractResponse(BaseModel):
    results: List[ExtractResult]
    failed_results: List[FailedResult]
    response_time: float
    request_id: str

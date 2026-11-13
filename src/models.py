from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PublishedAtTrust(str, Enum):
    PUBLISHER_CONFIRMED = "publisher_confirmed"
    PROVIDER_STAMPED = "provider_stamped"
    INFERRED = "inferred"
    MISSING_EXCLUDED = "missing_excluded"


class IngestionMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"
    HISTORICAL = "historical"
    BENCHMARK = "benchmark"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RawArticleEvent(BaseModel):
    schema_version: str = Field(default="1.0")
    article_id: str
    provider_event_id: Optional[str] = None
    publisher_id: str
    provider_id: str
    headline: str = Field(min_length=3)
    language: str = Field(default="en")
    article_url: Optional[str] = None
    tickers: List[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    published_at_trust: PublishedAtTrust = PublishedAtTrust.PUBLISHER_CONFIRMED
    provider_received_at: Optional[datetime] = None
    connector_received_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ingestion_mode: IngestionMode = IngestionMode.LIVE
    content_sha256: str = Field(default="")
    source_license_id: str = Field(default="access-agreement-reference")

    @model_validator(mode="before")
    @classmethod
    def populate_content_hash_and_id(cls, data: dict):
        if isinstance(data, dict):
            headline = data.get("headline", "").strip()
            if not data.get("content_sha256"):
                data["content_sha256"] = hashlib.sha256(headline.encode("utf-8")).hexdigest()
            if not data.get("article_id"):
                pub = data.get("publisher_id", "pub")
                prov_id = data.get("provider_event_id") or data["content_sha256"][:12]
                data["article_id"] = f"{pub}-{prov_id}"
            if not data.get("ingested_at"):
                data["ingested_at"] = datetime.now(timezone.utc)
        return data


class ScoredArticleEvent(BaseModel):
    article_id: str
    model_id: str = "ProsusAI/finbert"
    model_version: str = "main"
    preprocess_version: str = "v1"
    label: SentimentLabel
    p_positive: float = Field(ge=0.0, le=1.0)
    p_negative: float = Field(ge=0.0, le=1.0)
    p_neutral: float = Field(ge=0.0, le=1.0)
    sentiment_index: float = Field(ge=-1.0, le=1.0)
    cache_hit: bool = False
    inference_started_at: datetime
    score_completed_at: datetime
    mode: IngestionMode = IngestionMode.LIVE
    publication_to_score_ms: Optional[float] = None
    connector_to_score_ms: Optional[float] = None

    @field_validator("sentiment_index", mode="before")
    @classmethod
    def compute_sentiment_index(cls, v, info):
        # Allow passing None or explicit, default to p_positive - p_negative
        return v

    @model_validator(mode="after")
    def validate_probabilities(self) -> ScoredArticleEvent:
        total = self.p_positive + self.p_negative + self.p_neutral
        if abs(total - 1.0) > 0.05:
            # Re-normalize if slight floating point discrepancy
            norm = total if total > 0 else 1.0
            self.p_positive /= norm
            self.p_negative /= norm
            self.p_neutral /= norm
        self.sentiment_index = round(self.p_positive - self.p_negative, 4)
        return self


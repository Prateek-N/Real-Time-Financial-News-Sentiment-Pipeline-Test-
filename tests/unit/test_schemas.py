import json
import pytest
from datetime import datetime, timezone
from jsonschema import validate, ValidationError

from src.models import (
    IngestionMode,
    PublishedAtTrust,
    RawArticleEvent,
    ScoredArticleEvent,
    SentimentLabel,
)


def test_raw_article_event_valid():
    now = datetime.now(timezone.utc)
    event = RawArticleEvent(
        article_id="reuters-12345",
        provider_event_id="12345",
        publisher_id="reuters",
        provider_id="reuters_direct_feed",
        headline="Apple reports record quarterly revenue and raises guidance",
        tickers=["AAPL"],
        published_at=now,
        published_at_trust=PublishedAtTrust.PUBLISHER_CONFIRMED,
        ingested_at=now,
        ingestion_mode=IngestionMode.LIVE,
    )
    assert event.article_id == "reuters-12345"
    assert len(event.content_sha256) == 64
    assert event.tickers == ["AAPL"]


def test_raw_json_schema_validation():
    with open("schemas/raw_article.schema.json", "r") as f:
        schema = json.load(f)

    valid_payload = {
        "schema_version": "1.0",
        "article_id": "bloomberg-9988",
        "publisher_id": "bloomberg",
        "provider_id": "bloomberg_enterprise_api",
        "headline": "Fed holds interest rates steady amid controlled inflation",
        "published_at_trust": "publisher_confirmed",
        "ingested_at": "2026-11-13T12:00:00Z",
        "ingestion_mode": "live",
        "content_sha256": "a" * 64
    }
    # Should not raise exception
    validate(instance=valid_payload, schema=schema)

    # Missing mandatory field 'headline'
    invalid_payload = dict(valid_payload)
    del invalid_payload["headline"]
    with pytest.raises(ValidationError):
        validate(instance=invalid_payload, schema=schema)


def test_scored_article_event_computation():
    now = datetime.now(timezone.utc)
    scored = ScoredArticleEvent(
        article_id="reuters-12345",
        model_id="ProsusAI/finbert",
        model_version="main",
        preprocess_version="v1",
        label=SentimentLabel.POSITIVE,
        p_positive=0.85,
        p_negative=0.05,
        p_neutral=0.10,
        sentiment_index=0.80,
        cache_hit=False,
        inference_started_at=now,
        score_completed_at=now,
        mode=IngestionMode.LIVE
    )
    assert scored.sentiment_index == 0.80
    assert scored.label == SentimentLabel.POSITIVE
    assert abs((scored.p_positive + scored.p_negative + scored.p_neutral) - 1.0) < 1e-4


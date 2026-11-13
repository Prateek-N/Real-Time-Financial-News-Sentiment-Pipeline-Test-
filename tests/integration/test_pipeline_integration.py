import pytest
from datetime import datetime, timezone

from src.connectors.replay_connector import ReplayNewsConnector
from src.ml.cache import FinbertRedisCache
from src.ml.inference import FinbertInferenceEngine
from src.models import IngestionMode
from src.storage.db import DatabaseManager
from src.stream.broker import StreamBroker
from src.stream.consumer import InferenceStreamConsumer
from src.stream.producer import NewsStreamProducer


def test_full_pipeline_flow():
    # Setup test components
    broker = StreamBroker()
    cache = FinbertRedisCache(model_version="main", preprocess_version="v1")
    engine = FinbertInferenceEngine(cache=cache)
    db = DatabaseManager()
    producer = NewsStreamProducer(broker=broker)
    consumer = InferenceStreamConsumer(inference_engine=engine, db_manager=db, broker=broker)
    connector = ReplayNewsConnector()

    # 1. Produce valid event
    event = connector.generate_event(mode=IngestionMode.LIVE)
    pub_ok = producer.publish_raw_event(event)
    assert pub_ok

    # 2. Consume and score via consumer
    scored_count = consumer.process_microbatch(max_items=1, max_wait_ms=100)
    assert scored_count == 1

    # 3. Query DB and verify persistence
    rows = db.query_recent_scored_articles(limit=5)
    matching = [r for r in rows if r["article_id"] == event.article_id]
    assert len(matching) == 1
    scored_row = matching[0]

    assert scored_row["headline"] == event.headline
    assert scored_row["label"] in ["positive", "negative", "neutral"]
    assert -1.0 <= scored_row["sentiment_index"] <= 1.0

    # 4. Verify idempotent retry (re-persisting same article must NOT create duplicate rows)
    consumer.db.upsert_article(event)
    rows_after = db.query_recent_scored_articles(limit=10)
    matching_after = [r for r in rows_after if r["article_id"] == event.article_id]
    assert len(matching_after) == 1


def test_dead_letter_queue_on_malformed_event():
    broker = StreamBroker()
    cache = FinbertRedisCache()
    engine = FinbertInferenceEngine(cache=cache)
    db = DatabaseManager()
    consumer = InferenceStreamConsumer(inference_engine=engine, db_manager=db, broker=broker)

    # Publish completely malformed payload
    poison_payload = {"invalid_key": "not a valid article"}
    broker.publish("news.raw.v1", key="poison-1", value=poison_payload)

    # Consumer should route to DLQ without crashing
    consumer.process_microbatch(max_items=1, max_wait_ms=50)

    # Check DLQ topic has message
    dlq_messages = broker.consume_batch("news.dead_letter.v1", group_id="test-dlq", max_items=1)
    assert len(dlq_messages) == 1
    assert "raw_payload" in dlq_messages[0]["value"]


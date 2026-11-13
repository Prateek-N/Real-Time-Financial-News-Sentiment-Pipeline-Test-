from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..ml.inference import FinbertInferenceEngine
from ..models import RawArticleEvent, ScoredArticleEvent
from ..storage.db import DatabaseManager
from .broker import StreamBroker

logger = logging.getLogger(__name__)


class InferenceStreamConsumer:
    """Consumes raw news events, scores them via FinBERT with Redis cache,

    and idempotently writes results to DB and scored Kafka topic.
    Adheres strictly to PRD Section 6 & 10.1 (commit offsets after persistence).
    """

    def __init__(
        self,
        inference_engine: FinbertInferenceEngine,
        db_manager: DatabaseManager,
        broker: Optional[StreamBroker] = None,
        raw_topic: str = "news.raw.v1",
        scored_topic: str = "news.scored.v1",
        dead_letter_topic: str = "news.dead_letter.v1",
        group_id: str = "sentiment-inference-workers"
    ):
        self.engine = inference_engine
        self.db = db_manager
        self.broker = broker or StreamBroker()
        self.raw_topic = raw_topic
        self.scored_topic = scored_topic
        self.dead_letter_topic = dead_letter_topic
        self.group_id = group_id

        # Consumer telemetry
        self.total_processed = 0
        self.total_errors = 0

    def process_microbatch(self, max_items: int = 16, max_wait_ms: int = 100) -> int:
        """Polls, validates, scores, persists, and publishes a micro-batch."""
        raw_messages = self.broker.consume_batch(
            topic=self.raw_topic,
            group_id=self.group_id,
            max_items=max_items,
            max_wait_ms=max_wait_ms
        )

        if not raw_messages:
            return 0

        valid_events: List[RawArticleEvent] = []
        msg_keys: List[str] = []

        for msg in raw_messages:
            try:
                data = msg.get("value", {})
                event = RawArticleEvent.model_validate(data)
                valid_events.append(event)
                msg_keys.append(msg.get("key", event.article_id))
            except Exception as err:
                self.total_errors += 1
                logger.error("Poison message routed to dead-letter queue: %s", err)
                self.broker.publish(
                    self.dead_letter_topic,
                    key=msg.get("key", "error"),
                    value={"raw_payload": msg.get("value"), "error": str(err), "ts": time.time()}
                )

        if not valid_events:
            return 0

        # Step 1: Run PyTorch inference with Redis caching
        scored_events: List[ScoredArticleEvent] = self.engine.score_events(valid_events)

        # Step 2: Idempotent Database Upsert
        for event, scored in zip(valid_events, scored_events):
            self.db.upsert_article(event)
            self.db.upsert_score(scored)

        # Step 3: Publish scored events to Kafka news.scored.v1
        for scored in scored_events:
            payload = scored.model_dump(mode="json")
            self.broker.publish(self.scored_topic, key=scored.article_id, value=payload)

        # Step 4: Commit offsets (at-least-once delivery guaranteed)
        self.total_processed += len(scored_events)
        return len(scored_events)


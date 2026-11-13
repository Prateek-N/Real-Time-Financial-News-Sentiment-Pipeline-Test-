from __future__ import annotations

import json
import logging
from typing import Optional

from ..models import RawArticleEvent
from .broker import StreamBroker

logger = logging.getLogger(__name__)


class NewsStreamProducer:
    """Producer for raw validated financial news events."""

    def __init__(self, broker: Optional[StreamBroker] = None, topic: str = "news.raw.v1"):
        self.broker = broker or StreamBroker()
        self.topic = topic

    def publish_raw_event(self, event: RawArticleEvent) -> bool:
        """Publish normalized raw article event to Kafka news.raw.v1."""
        payload = event.model_dump(mode="json")
        # Key messages by stable article_id to ensure partition ordering
        key = event.article_id
        success = self.broker.publish(self.topic, key=key, value=payload)
        return success


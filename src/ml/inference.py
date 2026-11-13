from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..models import IngestionMode, PublishedAtTrust, RawArticleEvent, ScoredArticleEvent, SentimentLabel
from .cache import FinbertRedisCache

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "ProsusAI/finbert"
DEFAULT_REVISION = "main"


class FinbertInferenceEngine:
    """FinBERT PyTorch Online & Batch Inference Engine.

    Adheres strictly to PRD Section 10.1:
    - Loads model and tokenizer once per process.
    - Runs in model.eval() and torch.inference_mode().
    - Uses exact model.config.id2label mapping:
        0 -> positive, 1 -> negative, 2 -> neutral
    - Produces sentiment_index = p_positive - p_negative.
    - Integrates with model-versioned Redis cache.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = DEFAULT_REVISION,
        max_length: int = 128,
        device: Optional[str] = None,
        cache: Optional[FinbertRedisCache] = None,
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing FinBERT inference engine on %s (model=%s, revision=%s)", self.device, model_id, model_revision)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=model_revision)
        self.model.to(self.device)
        self.model.eval()

        # Validate label mapping from model config
        # Official ProsusAI/finbert: {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        logger.info("Loaded id2label mapping: %s", self.id2label)

        # Invert for fast lookup if needed
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.cache = cache
        self.preprocess_version = "v1"

    def predict_batch_raw(self, headlines: List[str]) -> List[Dict[str, Any]]:
        """Inference on raw list of headlines using PyTorch in torch.inference_mode()."""
        if not headlines:
            return []

        tokens = self.tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.inference_mode():
            outputs = self.model(**tokens)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: List[Dict[str, Any]] = []
        for i in range(len(headlines)):
            p_vec = probs[i]
            # Map probabilities according to model.config.id2label
            p_pos = float(p_vec[self.label2id.get("positive", 0)])
            p_neg = float(p_vec[self.label2id.get("negative", 1)])
            p_neu = float(p_vec[self.label2id.get("neutral", 2)])

            best_idx = int(p_vec.argmax())
            label_str = self.id2label.get(best_idx, "neutral")

            sentiment_idx = round(p_pos - p_neg, 4)

            results.append({
                "label": label_str,
                "p_positive": round(p_pos, 4),
                "p_negative": round(p_neg, 4),
                "p_neutral": round(p_neu, 4),
                "sentiment_index": sentiment_idx,
                "model_id": self.model_id,
                "model_version": self.model_revision,
                "preprocess_version": self.preprocess_version,
            })

        return results

    def score_event(self, event: RawArticleEvent) -> ScoredArticleEvent:
        """Score a single event with Redis cache lookup and timestamp tracking."""
        scored_events = self.score_events([event])
        return scored_events[0]

    def score_events(self, events: List[RawArticleEvent]) -> List[ScoredArticleEvent]:
        """Score micro-batch of events, utilizing Redis cache for unseen vs seen texts."""
        if not events:
            return []

        start_time = datetime.now(timezone.utc)
        headlines = [e.headline for e in events]
        n_events = len(events)

        scored_outputs: List[Optional[Dict[str, Any]]] = [None] * n_events
        cache_hits: List[bool] = [False] * n_events
        miss_indices: List[int] = list(range(n_events))

        # 1. Check Redis cache if configured
        if self.cache:
            cached_results, miss_indices = self.cache.mget(headlines)
            for idx, c_res in enumerate(cached_results):
                if c_res is not None:
                    scored_outputs[idx] = c_res
                    cache_hits[idx] = True

        # 2. Run PyTorch inference on cache misses
        if miss_indices:
            miss_headlines = [headlines[idx] for idx in miss_indices]
            inferred_results = self.predict_batch_raw(miss_headlines)

            # Store in cache and outputs
            cache_updates: List[Tuple[str, Dict[str, Any]]] = []
            for miss_idx, res in zip(miss_indices, inferred_results):
                scored_outputs[miss_idx] = res
                cache_updates.append((headlines[miss_idx], res))

            if self.cache and cache_updates:
                self.cache.mset(cache_updates)

        completed_time = datetime.now(timezone.utc)

        # 3. Construct ScoredArticleEvent objects with latency metrics
        final_events: List[ScoredArticleEvent] = []
        for i, event in enumerate(events):
            out = scored_outputs[i]
            label_val = SentimentLabel(out["label"])

            pub_to_score_ms: Optional[float] = None
            if event.published_at and event.published_at_trust == PublishedAtTrust.PUBLISHER_CONFIRMED:
                # PRD Section 14.2: score_completed_at - published_at
                delta = (completed_time - event.published_at).total_seconds() * 1000.0
                pub_to_score_ms = round(max(0.0, delta), 2)

            conn_to_score_ms: Optional[float] = None
            if event.connector_received_at:
                delta_c = (completed_time - event.connector_received_at).total_seconds() * 1000.0
                conn_to_score_ms = round(max(0.0, delta_c), 2)

            scored = ScoredArticleEvent(
                article_id=event.article_id,
                model_id=self.model_id,
                model_version=self.model_revision,
                preprocess_version=self.preprocess_version,
                label=label_val,
                p_positive=out["p_positive"],
                p_negative=out["p_negative"],
                p_neutral=out["p_neutral"],
                sentiment_index=out["sentiment_index"],
                cache_hit=cache_hits[i],
                inference_started_at=start_time,
                score_completed_at=completed_time,
                mode=event.ingestion_mode,
                publication_to_score_ms=pub_to_score_ms,
                connector_to_score_ms=conn_to_score_ms,
            )
            final_events.append(scored)

        return final_events


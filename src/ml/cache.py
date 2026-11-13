from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_headline_hash(headline: str) -> str:
    """Computes SHA-256 of lowercase, normalized headline, stripping report ID suffixes."""
    norm = re.sub(r"\s*\[(?:Report|update)\s*#\d+\]", "", headline, flags=re.IGNORECASE).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class FinbertRedisCache:
    """Model-versioned Redis cache for FinBERT sentiment outputs.

    Cache Key: finbert:{model_checkpoint_sha}:{preprocess_version}:{sha256(normalized_headline)}
    Tracks cold vs warm hits/misses per PRD Section 10.2.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 30 * 86400,
        model_version: str = "main",
        preprocess_version: str = "v1",
        enable_fallback: bool = True
    ):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.model_version = model_version
        self.preprocess_version = preprocess_version
        self.enable_fallback = enable_fallback

        self.client = None
        self._memory_fallback: Dict[str, str] = {}
        self.using_fallback = False

        # Hit/Miss accounting
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self._connect()

    def _connect(self):
        try:
            import redis
            client = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=1.5)
            client.ping()
            self.client = client
            self.using_fallback = False
            logger.info("Connected to Redis at %s", self.redis_url)
        except Exception as e:
            if self.enable_fallback:
                try:
                    import fakeredis
                    self.client = fakeredis.FakeRedis(decode_responses=True)
                    self.using_fallback = True
                    logger.warning("Redis server unreachable (%s). Using embedded fakeredis.", e)
                except Exception:
                    self.client = None
                    self.using_fallback = True
                    logger.warning("Redis and fakeredis unavailable. Using in-memory dict cache.")
            else:
                raise

    def make_key(self, headline: str) -> str:
        h_hash = compute_headline_hash(headline)
        return f"finbert:{self.model_version}:{self.preprocess_version}:{h_hash}"

    def get(self, headline: str) -> Optional[Dict[str, Any]]:
        self.total_lookups += 1
        key = self.make_key(headline)
        try:
            if self.client:
                data = self.client.get(key)
            else:
                data = self._memory_fallback.get(key)

            if data:
                self.cache_hits += 1
                return json.loads(data)
            else:
                self.cache_misses += 1
                return None
        except Exception as err:
            logger.error("Redis get failed for key %s: %s", key, err)
            self.cache_misses += 1
            return None

    def mget(self, headlines: List[str]) -> Tuple[List[Optional[Dict[str, Any]]], List[int]]:
        """Multi-get for micro-batching.

        Returns:
            results: List of cached results (or None for misses)
            miss_indices: Indices of headlines that missed the cache
        """
        if not headlines:
            return [], []

        keys = [self.make_key(h) for h in headlines]
        self.total_lookups += len(headlines)

        raw_values: List[Optional[str]] = []
        try:
            if self.client:
                raw_values = self.client.mget(keys)
            else:
                raw_values = [self._memory_fallback.get(k) for k in keys]
        except Exception as err:
            logger.error("Redis mget failed: %s", err)
            raw_values = [None] * len(headlines)

        results: List[Optional[Dict[str, Any]]] = []
        miss_indices: List[int] = []

        for idx, val in enumerate(raw_values):
            if val is not None:
                self.cache_hits += 1
                try:
                    results.append(json.loads(val))
                except Exception:
                    results.append(None)
                    miss_indices.append(idx)
            else:
                self.cache_misses += 1
                results.append(None)
                miss_indices.append(idx)

        return results, miss_indices

    def set(self, headline: str, score_payload: Dict[str, Any]):
        key = self.make_key(headline)
        payload = dict(score_payload)
        payload["cached_at_ts"] = time.time()
        json_val = json.dumps(payload)
        try:
            if self.client:
                self.client.set(key, json_val, ex=self.ttl_seconds)
            else:
                self._memory_fallback[key] = json_val
        except Exception as err:
            logger.error("Redis set failed for %s: %s", key, err)

    def mset(self, headline_payload_pairs: List[Tuple[str, Dict[str, Any]]]):
        if not headline_payload_pairs:
            return
        now = time.time()
        kv_pairs = {}
        for h, payload in headline_payload_pairs:
            key = self.make_key(h)
            p = dict(payload)
            p["cached_at_ts"] = now
            kv_pairs[key] = json.dumps(p)

        try:
            if self.client:
                pipe = self.client.pipeline()
                for k, v in kv_pairs.items():
                    pipe.set(k, v, ex=self.ttl_seconds)
                pipe.execute()
            else:
                self._memory_fallback.update(kv_pairs)
        except Exception as err:
            logger.error("Redis mset failed: %s", err)

    def stats(self) -> Dict[str, Any]:
        hit_rate = (self.cache_hits / self.total_lookups) if self.total_lookups > 0 else 0.0
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 4),
            "fallback_active": self.using_fallback,
            "model_version": self.model_version,
            "preprocess_version": self.preprocess_version
        }

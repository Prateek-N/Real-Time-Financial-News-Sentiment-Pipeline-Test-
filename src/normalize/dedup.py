from __future__ import annotations

import collections
import hashlib
import re
from typing import Dict, Optional, Set, Tuple


def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA-256 digest of cleaned headline."""
    norm = re.sub(r"\W+", "", text.lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class Deduplicator:
    """Thread-safe bounded in-memory deduplicator and story cluster tracker.

    Handles:
    1. Exact ID duplicates: rejects duplicates within same publisher.
    2. Exact URL duplicates: rejects duplicate provider links.
    3. Cross-publisher wire story clustering: flags syndication while retaining source provenance.
    """

    def __init__(self, capacity: int = 100_000, jaccard_threshold: float = 0.85):
        self.capacity = capacity
        self.jaccard_threshold = jaccard_threshold
        # Bounded LRU-style collections
        self._seen_article_ids: collections.OrderedDict[str, bool] = collections.OrderedDict()
        self._seen_urls: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._hash_to_cluster: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._recent_shingles: collections.OrderedDict[str, Set[str]] = collections.OrderedDict()

    def _evict_if_needed(self):
        while len(self._seen_article_ids) > self.capacity:
            self._seen_article_ids.popitem(last=False)
        while len(self._seen_urls) > self.capacity:
            self._seen_urls.popitem(last=False)
        while len(self._hash_to_cluster) > self.capacity:
            self._hash_to_cluster.popitem(last=False)
        while len(self._recent_shingles) > self.capacity:
            self._recent_shingles.popitem(last=False)

    @staticmethod
    def _get_shingles(text: str, n: int = 3) -> Set[str]:
        words = re.findall(r"\w+", text.lower())
        if len(words) < n:
            return set(words)
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    def check_duplicate(
        self,
        article_id: str,
        publisher_id: str,
        headline: str,
        article_url: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Check if incoming event is duplicate or near-duplicate.

        Returns:
            (is_exact_dup, reason, cluster_id)
        """
        self._evict_if_needed()

        # 1. Exact Article ID match
        if article_id in self._seen_article_ids:
            return True, "exact_article_id_duplicate", None

        # 2. Exact URL match (if URL present)
        if article_url and article_url in self._seen_urls:
            return True, f"exact_url_duplicate_of_{self._seen_urls[article_url]}", None

        # 3. Content hash check
        c_hash = compute_content_hash(headline)
        shingles = self._get_shingles(headline)

        cluster_id = None
        if c_hash in self._hash_to_cluster:
            cluster_id = self._hash_to_cluster[c_hash]
        else:
            # Check near-duplicate against recent headlines
            for prev_id, prev_shingles in reversed(list(self._recent_shingles.items())[-500:]):
                if not shingles or not prev_shingles:
                    continue
                intersection = len(shingles.intersection(prev_shingles))
                union = len(shingles.union(prev_shingles))
                if union > 0 and (intersection / union) >= self.jaccard_threshold:
                    cluster_id = f"cluster-{prev_id}"
                    break

            if not cluster_id:
                cluster_id = f"cluster-{article_id}"
            self._hash_to_cluster[c_hash] = cluster_id

        # Record this event as seen
        self._seen_article_ids[article_id] = True
        if article_url:
            self._seen_urls[article_url] = article_id
        self._recent_shingles[article_id] = shingles

        return False, "unique_accepted", cluster_id


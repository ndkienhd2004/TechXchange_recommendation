from __future__ import annotations

import datetime as dt
import threading

from app.config import Settings
from app.embedding_cache import EmbeddingCache
from app.gemini_embedding_client import GeminiEmbeddingClient
from app.recommendation_builders import (
    build_cf_item_neighbors,
    build_item_embeddings,
    build_item_meta,
    build_popularity_map,
    build_user_event_features,
)
from app.recommendation_constants import EVENT_WEIGHTS
from app.recommendation_scorers import (
    recommend_collaborative_cached,
    recommend_content_based_cached,
    similar_products_collaborative,
    similar_products_content,
)
from app.repositories import RecommendationRepository


class RecommendationService:
    def __init__(self, repo: RecommendationRepository, settings: Settings) -> None:
        self.repo = repo
        self.settings = settings

        self.item_vectors: dict[int, list[float]] = {}
        self.item_meta: dict[int, dict] = {}
        self.popularity: dict[int, float] = {}

        self.user_profiles: dict[int, dict] = {}
        self.user_item_scores: dict[int, dict[int, float]] = {}
        self.user_seen_products: dict[int, set[int]] = {}
        self.cf_item_neighbors: dict[int, list[tuple[int, float]]] = {}

        self.last_refreshed_at: dt.datetime | None = None
        self._refresh_lock = threading.Lock()

        self.embedding_cache = EmbeddingCache(settings.embedding_cache_path)
        self.embedding_client = GeminiEmbeddingClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embed_model,
            output_dimensionality=settings.gemini_embed_dim,
            timeout_seconds=settings.gemini_timeout_seconds,
            batch_size=settings.gemini_batch_size,
            use_vertexai=settings.google_genai_use_vertexai,
            google_cloud_project=settings.google_cloud_project,
            google_cloud_location=settings.google_cloud_location,
        )

    def ensure_fresh(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        if self.last_refreshed_at is None:
            self.refresh()
            return
        elapsed = now - self.last_refreshed_at
        if elapsed.total_seconds() >= self.settings.refresh_interval_minutes * 60:
            self.refresh()

    def refresh(self) -> None:
        with self._refresh_lock:
            products = self.repo.fetch_active_products()
            self.item_meta = build_item_meta(products)
            self.item_vectors = build_item_embeddings(
                products=products,
                settings=self.settings,
                embedding_cache=self.embedding_cache,
                embedding_client=self.embedding_client,
            )
            self.popularity = build_popularity_map(
                repo=self.repo,
                lookback_days=self.settings.popularity_lookback_days,
            )

            all_user_events = self.repo.fetch_user_events_all(self.settings.event_lookback_days)
            (
                self.user_profiles,
                self.user_item_scores,
                self.user_seen_products,
            ) = build_user_event_features(
                event_rows=all_user_events,
                item_vectors=self.item_vectors,
                event_weights=EVENT_WEIGHTS,
            )
            self.cf_item_neighbors = build_cf_item_neighbors(
                user_item_scores=self.user_item_scores,
                item_ids=list(self.item_meta.keys()),
                min_similarity=self.settings.cf_min_similarity,
                max_neighbors=self.settings.cf_max_neighbors,
            )
            self.last_refreshed_at = dt.datetime.now(dt.timezone.utc)

    def recommend_for_user(self, user_id: int, limit: int = 10) -> dict:
        self.ensure_fresh()
        items = recommend_content_based_cached(
            user_id=user_id,
            limit=limit,
            user_profiles=self.user_profiles,
            item_vectors=self.item_vectors,
            item_meta=self.item_meta,
            popularity=self.popularity,
        )
        strategy = (
            "content_based_gemini_embedding"
            if items and items[0].get("content_score", 0) > 0
            else "content_based_fallback_popular"
        )
        return {"strategy": strategy, "user_id": user_id, "items": items, "meta": self._meta()}

    def recommend_collaborative_for_user(self, user_id: int, limit: int = 10) -> dict:
        self.ensure_fresh()
        items = recommend_collaborative_cached(
            user_id=user_id,
            limit=limit,
            user_item_scores=self.user_item_scores,
            user_seen_products=self.user_seen_products,
            cf_item_neighbors=self.cf_item_neighbors,
            item_meta=self.item_meta,
            popularity=self.popularity,
        )
        strategy = (
            "collaborative_filtering_item_item"
            if items and items[0].get("collab_score", 0) > 0
            else "collaborative_fallback_popular"
        )
        return {"strategy": strategy, "user_id": user_id, "items": items, "meta": self._meta()}

    def recommend_hybrid_for_user(self, user_id: int, limit: int = 10) -> dict:
        self.ensure_fresh()
        limit = max(1, min(limit, 100))
        content_rows = recommend_content_based_cached(
            user_id=user_id,
            limit=max(limit * 3, 30),
            user_profiles=self.user_profiles,
            item_vectors=self.item_vectors,
            item_meta=self.item_meta,
            popularity=self.popularity,
        )
        collab_rows = recommend_collaborative_cached(
            user_id=user_id,
            limit=max(limit * 3, 30),
            user_item_scores=self.user_item_scores,
            user_seen_products=self.user_seen_products,
            cf_item_neighbors=self.cf_item_neighbors,
            item_meta=self.item_meta,
            popularity=self.popularity,
        )

        alpha = min(1.0, max(0.0, self.settings.hybrid_content_weight))
        if user_id not in self.user_item_scores:
            alpha = 1.0

        content_map = {int(row["product_id"]): float(row.get("score", 0.0)) for row in content_rows}
        collab_map = {int(row["product_id"]): float(row.get("score", 0.0)) for row in collab_rows}
        all_ids = set(content_map.keys()) | set(collab_map.keys())

        seen_products = self.user_seen_products.get(user_id, set())
        unseen_rows: list[dict] = []
        seen_rows: list[dict] = []
        for pid in all_ids:
            content_score = content_map.get(pid, 0.0)
            collab_score = collab_map.get(pid, 0.0)
            hybrid_score = (alpha * content_score) + ((1.0 - alpha) * collab_score)
            if hybrid_score <= 0:
                continue
            meta = self.item_meta.get(pid, {})
            row = {
                "product_id": pid,
                "name": meta.get("name"),
                "price": meta.get("price"),
                "category_id": meta.get("category_id"),
                "brand_id": meta.get("brand_id"),
                "score": round(hybrid_score, 6),
                "content_component": round(content_score, 6),
                "collaborative_component": round(collab_score, 6),
            }
            if pid in seen_products:
                seen_rows.append(row)
            else:
                unseen_rows.append(row)

        unseen_rows.sort(key=lambda x: x["score"], reverse=True)
        seen_rows.sort(key=lambda x: x["score"], reverse=True)
        rows = unseen_rows[:limit]
        if len(rows) < limit:
            rows.extend(seen_rows[: limit - len(rows)])
        return {
            "strategy": "hybrid_content_collaborative",
            "user_id": user_id,
            "items": rows,
            "meta": {**self._meta(), "hybrid_content_weight": alpha},
        }

    def similar_products(self, product_id: int, limit: int = 10) -> dict:
        self.ensure_fresh()
        rows = similar_products_content(
            product_id=product_id,
            limit=limit,
            item_vectors=self.item_vectors,
            item_meta=self.item_meta,
        )
        return {
            "strategy": "content_based_item_to_item_gemini_embedding",
            "product_id": product_id,
            "items": rows,
            "meta": self._meta(),
        }

    def similar_products_collaborative(self, product_id: int, limit: int = 10) -> dict:
        self.ensure_fresh()
        rows = similar_products_collaborative(
            product_id=product_id,
            limit=limit,
            cf_item_neighbors=self.cf_item_neighbors,
            item_meta=self.item_meta,
        )
        return {
            "strategy": "collaborative_item_to_item",
            "product_id": product_id,
            "items": rows,
            "meta": self._meta(),
        }

    def _meta(self) -> dict:
        cf_items_with_neighbors = sum(1 for _pid, lst in self.cf_item_neighbors.items() if lst)
        return {
            "active_items": len(self.item_vectors),
            "cached_users": len(self.user_profiles),
            "cf_users": len(self.user_item_scores),
            "cf_items_with_neighbors": cf_items_with_neighbors,
            "refresh_interval_minutes": self.settings.refresh_interval_minutes,
            "last_refreshed_at": (
                self.last_refreshed_at.isoformat() if self.last_refreshed_at else None
            ),
            "embedding_cache": self.embedding_cache.stats(),
            "embedding_model": self.settings.gemini_embed_model,
        }


class ContentBasedRecommender(RecommendationService):
    """Backward-compatible alias for old class name."""


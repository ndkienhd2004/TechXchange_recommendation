from __future__ import annotations

import datetime as dt
import math
import threading
from collections import defaultdict

from app.config import Settings
from app.embedding_cache import EmbeddingCache
from app.gemini_embedding_client import GeminiEmbeddingClient
from app.repositories import RecommendationRepository
from app.vector_utils import add_scaled_dense, cosine_dense, normalize_dense

EVENT_WEIGHTS: dict[str, float] = {
    "impression": 1.0,
    "view": 2.0,
    "click": 3.0,
    "add_to_cart": 6.0,
    "wishlist": 7.0,
    "purchase": 12.0,
}


class ContentBasedRecommender:
    def __init__(self, repo: RecommendationRepository, settings: Settings) -> None:
        """Recommender trung tâm: content-based + collaborative + hybrid.

        State chính được build bằng `refresh()` và giữ trong memory:
        - `item_vectors`: embedding vector theo product_id
        - `item_meta`: metadata tối thiểu để trả về response nhanh
        - `popularity`: pop_score theo product_id
        - `user_profiles`: content profile vector theo user_id (từ user events)
        - `user_item_scores`, `user_seen_products`: implicit feedback cho collaborative
        - `cf_item_neighbors`: danh sách hàng xóm item-item theo cosine similarity trong không gian implicit

        Nguồn dữ liệu:
        - Products + events lấy từ `RecommendationRepository` (Postgres)
        - Embedding lấy từ Gemini (có cache file để giảm chi phí và tăng tốc)
        """
        self.repo = repo
        self.settings = settings

        self.item_vectors: dict[int, list[float]] = {}
        self.item_meta: dict[int, dict] = {}
        self.popularity: dict[int, float] = {}

        # Content profiles derived from embedding vectors
        self.user_profiles: dict[int, dict] = {}

        # Collaborative filtering state
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
        )

    def ensure_fresh(self) -> None:
        """Đảm bảo cache in-memory còn mới theo TTL (`refresh_interval_minutes`)."""
        now = dt.datetime.now(dt.timezone.utc)
        if self.last_refreshed_at is None:
            self.refresh()
            return
        elapsed = now - self.last_refreshed_at
        if elapsed.total_seconds() >= self.settings.refresh_interval_minutes * 60:
            self.refresh()

    def refresh(self) -> None:
        """Rebuild toàn bộ state recommender (thread-safe nhờ `_refresh_lock`).

        Thứ tự build:
        - Load products active → meta + embeddings
        - Tính popularity theo events lookback
        - Load user events aggregates → build user profiles + implicit feedback matrices
        - Build CF item neighbors (item-item) từ implicit feedback
        """
        with self._refresh_lock:
            products = self.repo.fetch_active_products()
            self.item_meta = self._build_item_meta(products)
            self.item_vectors = self._build_item_embeddings(products)
            self.popularity = self._build_popularity_map()

            all_user_events = self.repo.fetch_user_events_all(self.settings.event_lookback_days)
            (
                self.user_profiles,
                self.user_item_scores,
                self.user_seen_products,
            ) = self._build_user_event_features(all_user_events)
            self.cf_item_neighbors = self._build_cf_item_neighbors(self.user_item_scores)
            self.last_refreshed_at = dt.datetime.now(dt.timezone.utc)

    # Public APIs
    def recommend_for_user(self, user_id: int, limit: int = 10) -> dict:
        """Recommend content-based cho user_id.

        - Nếu có user profile vector → tính similarity với từng item_vector và blend nhẹ popularity.
        - Nếu cold-start → fallback theo popularity.
        """
        self.ensure_fresh()
        items = self._recommend_content_based_cached(user_id=user_id, limit=limit)
        strategy = (
            "content_based_gemini_embedding"
            if items and items[0].get("content_score", 0) > 0
            else "content_based_fallback_popular"
        )
        return {"strategy": strategy, "user_id": user_id, "items": items, "meta": self._meta()}

    def recommend_collaborative_for_user(self, user_id: int, limit: int = 10) -> dict:
        """Recommend collaborative filtering cho user_id (item-item CF).

        Dựa trên lịch sử event (implicit feedback) để:
        - lấy hàng xóm của các item user đã tương tác
        - tổng hợp score theo similarity * interaction_score
        - normalize và blend nhẹ popularity
        """
        self.ensure_fresh()
        items = self._recommend_collaborative_cached(user_id=user_id, limit=limit)
        strategy = (
            "collaborative_filtering_item_item"
            if items and items[0].get("collab_score", 0) > 0
            else "collaborative_fallback_popular"
        )
        return {"strategy": strategy, "user_id": user_id, "items": items, "meta": self._meta()}

    def recommend_hybrid_for_user(self, user_id: int, limit: int = 10) -> dict:
        """Recommend hybrid = blend content-based và collaborative.

        `alpha = hybrid_content_weight` (clamp 0..1).
        Nếu user không có collaborative state → alpha forced = 1.0 (pure content/popularity).
        """
        self.ensure_fresh()
        limit = max(1, min(limit, 100))
        content_rows = self._recommend_content_based_cached(user_id, max(limit * 3, 30))
        collab_rows = self._recommend_collaborative_cached(user_id, max(limit * 3, 30))

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
        """Item-to-item similarity theo content embedding vector (cosine + heuristic boost)."""
        self.ensure_fresh()
        limit = max(1, min(limit, 100))

        base_vector = self.item_vectors.get(product_id)
        base_meta = self.item_meta.get(product_id)
        if not base_vector or not base_meta:
            raise ValueError(f"Product {product_id} not found or not active")

        rows: list[dict] = []
        for other_id, other_vec in self.item_vectors.items():
            if other_id == product_id:
                continue
            content_score = cosine_dense(base_vector, other_vec)
            if content_score <= 0:
                continue
            other_meta = self.item_meta.get(other_id, {})

            boost = 0.0
            if other_meta.get("category_id") == base_meta.get("category_id"):
                boost += 0.05
            if base_meta.get("brand_id") and (
                other_meta.get("brand_id") == base_meta.get("brand_id")
            ):
                boost += 0.03

            final_score = content_score + boost
            rows.append(
                {
                    "product_id": other_id,
                    "name": other_meta.get("name"),
                    "price": other_meta.get("price"),
                    "category_id": other_meta.get("category_id"),
                    "brand_id": other_meta.get("brand_id"),
                    "score": round(final_score, 6),
                    "content_score": round(content_score, 6),
                }
            )

        rows.sort(key=lambda x: x["score"], reverse=True)
        return {
            "strategy": "content_based_item_to_item_gemini_embedding",
            "product_id": product_id,
            "items": rows[:limit],
            "meta": self._meta(),
        }

    def similar_products_collaborative(self, product_id: int, limit: int = 10) -> dict:
        """Item-to-item similarity theo collaborative neighbors (precomputed)."""
        self.ensure_fresh()
        limit = max(1, min(limit, 100))
        if product_id not in self.item_meta:
            raise ValueError(f"Product {product_id} not found or not active")

        neighbors = self.cf_item_neighbors.get(product_id, [])
        rows: list[dict] = []
        for other_id, sim in neighbors[:limit]:
            meta = self.item_meta.get(other_id, {})
            rows.append(
                {
                    "product_id": other_id,
                    "name": meta.get("name"),
                    "price": meta.get("price"),
                    "category_id": meta.get("category_id"),
                    "brand_id": meta.get("brand_id"),
                    "score": round(sim, 6),
                    "collab_score": round(sim, 6),
                }
            )
        return {
            "strategy": "collaborative_item_to_item",
            "product_id": product_id,
            "items": rows,
            "meta": self._meta(),
        }

    # Internal content/collab scoring
    def _recommend_content_based_cached(self, user_id: int, limit: int) -> list[dict]:
        """Tính list recommendation content-based.

        Ưu tiên item chưa seen; nếu chưa đủ limit thì backfill bằng item đã seen.
        """
        limit = max(1, min(limit, 100))
        profile = self.user_profiles.get(user_id)
        if not profile or not profile.get("vector"):
            return self._fallback_popular(limit)

        user_vector = profile["vector"]
        seen_products = profile["seen_products"]
        pop_max = max(self.popularity.values()) if self.popularity else 0.0

        unseen_rows: list[dict] = []
        seen_rows: list[dict] = []
        for product_id, item_vector in self.item_vectors.items():
            content_score = cosine_dense(user_vector, item_vector)
            if content_score <= 0:
                continue
            pop = self.popularity.get(product_id, 0.0)
            pop_norm = (pop / pop_max) if pop_max > 0 else 0.0
            final_score = (content_score * 0.9) + (pop_norm * 0.1)
            meta = self.item_meta.get(product_id, {})
            row = {
                "product_id": product_id,
                "name": meta.get("name"),
                "price": meta.get("price"),
                "category_id": meta.get("category_id"),
                "brand_id": meta.get("brand_id"),
                "score": round(final_score, 6),
                "content_score": round(content_score, 6),
                "popularity_boost": round(pop_norm, 6),
            }
            if product_id in seen_products:
                seen_rows.append(row)
            else:
                unseen_rows.append(row)
        unseen_rows.sort(key=lambda x: x["score"], reverse=True)
        seen_rows.sort(key=lambda x: x["score"], reverse=True)
        rows = unseen_rows[:limit]
        if len(rows) < limit:
            rows.extend(seen_rows[: limit - len(rows)])
        return rows

    def _recommend_collaborative_cached(self, user_id: int, limit: int) -> list[dict]:
        """Tính list recommendation collaborative (item-item) cho 1 user.

        Ưu tiên item chưa seen; nếu chưa đủ limit thì backfill bằng item đã seen.
        """
        limit = max(1, min(limit, 100))
        user_scores = self.user_item_scores.get(user_id)
        if not user_scores:
            return self._fallback_popular(limit)

        seen_products = self.user_seen_products.get(user_id, set())
        raw_scores: dict[int, float] = defaultdict(float)
        sim_sums: dict[int, float] = defaultdict(float)

        for item_id, interaction_score in user_scores.items():
            for neighbor_id, sim in self.cf_item_neighbors.get(item_id, []):
                raw_scores[neighbor_id] += sim * interaction_score
                sim_sums[neighbor_id] += abs(sim)

        if not raw_scores:
            return self._fallback_popular(limit)

        collab_scores: dict[int, float] = {}
        for pid, val in raw_scores.items():
            denom = sim_sums.get(pid, 0.0)
            collab_scores[pid] = (val / denom) if denom > 0 else val

        max_collab = max(collab_scores.values()) if collab_scores else 0.0
        pop_max = max(self.popularity.values()) if self.popularity else 0.0

        unseen_rows: list[dict] = []
        seen_rows: list[dict] = []
        for pid, val in collab_scores.items():
            collab_norm = (val / max_collab) if max_collab > 0 else 0.0
            pop_norm = (self.popularity.get(pid, 0.0) / pop_max) if pop_max > 0 else 0.0
            final_score = (collab_norm * 0.95) + (pop_norm * 0.05)
            meta = self.item_meta.get(pid, {})
            row = {
                "product_id": pid,
                "name": meta.get("name"),
                "price": meta.get("price"),
                "category_id": meta.get("category_id"),
                "brand_id": meta.get("brand_id"),
                "score": round(final_score, 6),
                "collab_score": round(collab_norm, 6),
                "popularity_boost": round(pop_norm, 6),
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
        return rows

    # Build/cache helpers
    def _build_item_meta(self, products: list[dict]) -> dict[int, dict]:
        """Chuyển product rows thành meta tối thiểu cho response nhanh."""
        meta: dict[int, dict] = {}
        for row in products:
            pid = int(row["id"])
            meta[pid] = {
                "id": pid,
                "name": row.get("name"),
                "price": float(row["price"]) if row.get("price") is not None else None,
                "category_id": row.get("category_id"),
                "brand_id": row.get("brand_id"),
            }
        return meta

    def _build_item_embeddings(self, products: list[dict]) -> dict[int, list[float]]:
        """Build embedding vectors cho product active, dùng cache để tránh gọi API lại.

        Cache key = (product_id, content_hash(text)).
        Nếu có cache-miss và không có `GEMINI_API_KEY` → raise để báo cấu hình sai.
        """
        if not products:
            return {}

        texts: dict[int, str] = {}
        for row in products:
            pid = int(row["id"])
            texts[pid] = self._product_to_embedding_text(row)

        self.embedding_cache.load()
        self.embedding_cache.ensure_model(self.settings.gemini_embed_model)
        vectors: dict[int, list[float]] = {}
        to_embed: list[tuple[int, str, str]] = []

        for pid, text in texts.items():
            content_hash = self.embedding_cache.build_hash(text)
            cached = self.embedding_cache.get(pid, content_hash)
            if cached:
                vectors[pid] = normalize_dense(cached)
                continue
            to_embed.append((pid, text, content_hash))

        if to_embed:
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required to build embeddings")
            embed_texts = [x[1] for x in to_embed]
            new_vectors = self.embedding_client.embed_texts(embed_texts)
            if len(new_vectors) != len(to_embed):
                raise RuntimeError(
                    f"Embedding size mismatch: expected {len(to_embed)}, got {len(new_vectors)}"
                )
            for idx, (pid, _text, content_hash) in enumerate(to_embed):
                vec = normalize_dense(new_vectors[idx])
                self.embedding_cache.set(pid, content_hash, vec)
                vectors[pid] = vec

        self.embedding_cache.delete_missing(set(texts.keys()))
        self.embedding_cache.save(self.settings.gemini_embed_model)
        return vectors

    def _product_to_embedding_text(self, row: dict) -> str:
        """Serialize metadata product thành text ổn định để embed."""
        parts = [
            f"name: {row.get('name') or ''}",
            f"description: {row.get('description') or ''}",
            f"catalog_name: {row.get('catalog_name') or ''}",
            f"catalog_description: {row.get('catalog_description') or ''}",
            f"catalog_specs: {row.get('catalog_specs') or ''}",
            f"brand_name: {row.get('brand_name') or ''}",
            f"category_name: {row.get('category_name') or ''}",
            f"quality: {row.get('quality') or ''}",
        ]
        return "\n".join(parts)

    def _build_popularity_map(self) -> dict[int, float]:
        """Build map product_id -> pop_score từ DB (weighted sum event types)."""
        rows = self.repo.fetch_product_popularity(self.settings.popularity_lookback_days)
        output: dict[int, float] = {}
        for row in rows:
            pid = int(row["product_id"])
            output[pid] = float(row["pop_score"] or 0.0)
        return output

    def _fallback_popular(self, limit: int) -> list[dict]:
        """Fallback recommendation: trả top popular trong active items."""
        pop_max = max(self.popularity.values()) if self.popularity else 0.0
        rows: list[dict] = []
        for product_id, meta in self.item_meta.items():
            pop = self.popularity.get(product_id, 0.0)
            pop_norm = (pop / pop_max) if pop_max > 0 else 0.0
            rows.append(
                {
                    "product_id": product_id,
                    "name": meta.get("name"),
                    "price": meta.get("price"),
                    "category_id": meta.get("category_id"),
                    "brand_id": meta.get("brand_id"),
                    "score": round(pop_norm, 6),
                    "content_score": 0.0,
                    "collab_score": 0.0,
                    "popularity_boost": round(pop_norm, 6),
                }
            )
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:limit]

    def _build_user_event_features(
        self, event_rows: list[dict]
    ) -> tuple[dict[int, dict], dict[int, dict[int, float]], dict[int, set[int]]]:
        """Chuyển events aggregate thành:

        - content profile vector cho từng user (weighted average embedding các product đã tương tác)
        - implicit feedback score matrix `user_item_scores[user_id][product_id]`
        - set item đã seen theo user để lọc khi recommend

        Score mỗi event:
        - base_weight = EVENT_WEIGHTS[type] * max(count, 1)
        - recency decay = exp(-age_days/30)
        - score = base_weight * recency
        """
        now = dt.datetime.now(dt.timezone.utc)
        weighted: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        seen_products: dict[int, set[int]] = defaultdict(set)

        for row in event_rows:
            user_id = int(row["user_id"])
            product_id = int(row["product_id"])
            if product_id not in self.item_vectors:
                continue

            event_type = str(row["event_type"])
            event_count = int(row["event_count"] or 0)
            base_weight = EVENT_WEIGHTS.get(event_type, 0.0) * max(event_count, 1)

            last_at = row.get("last_event_at")
            recency = 1.0
            if isinstance(last_at, dt.datetime):
                if last_at.tzinfo is None:
                    last_at = last_at.replace(tzinfo=dt.timezone.utc)
                age_days = (now - last_at).total_seconds() / 86400.0
                recency = math.exp(-age_days / 30.0)

            score = base_weight * recency
            if score <= 0:
                continue

            weighted[user_id][product_id] += score
            seen_products[user_id].add(product_id)

        content_profiles: dict[int, dict] = {}
        for uid, product_weight in weighted.items():
            total_weight = sum(product_weight.values())
            if total_weight <= 0:
                continue
            vector: list[float] = []
            for pid, weight in product_weight.items():
                scale = weight / total_weight
                vector = add_scaled_dense(vector, self.item_vectors[pid], scale)
            content_profiles[uid] = {
                "vector": normalize_dense(vector),
                "seen_products": seen_products.get(uid, set()),
            }

        # Convert defaultdict to plain dict for stable behavior
        user_item_scores = {uid: dict(items) for uid, items in weighted.items()}
        user_seen = {uid: set(items) for uid, items in seen_products.items()}
        return content_profiles, user_item_scores, user_seen

    def _build_cf_item_neighbors(
        self, user_item_scores: dict[int, dict[int, float]]
    ) -> dict[int, list[tuple[int, float]]]:
        """Build item-item neighbors dựa trên implicit feedback.

        Công thức:
        - Mỗi item là 1 vector trong không gian user interactions (sparse).
        - Tính dot product cho các cặp item cùng xuất hiện trong 1 user.
        - Similarity ~ cosine(dot / (||a||*||b||)).
        - Lọc theo `cf_min_similarity` và cắt top `cf_max_neighbors`.
        """
        item_norm_sq: dict[int, float] = defaultdict(float)
        pair_dot: dict[tuple[int, int], float] = defaultdict(float)

        for _uid, item_scores in user_item_scores.items():
            pairs = list(item_scores.items())
            for item_id, score in pairs:
                item_norm_sq[item_id] += score * score
            for i in range(len(pairs)):
                item_i, score_i = pairs[i]
                for j in range(i + 1, len(pairs)):
                    item_j, score_j = pairs[j]
                    a, b = (item_i, item_j) if item_i < item_j else (item_j, item_i)
                    pair_dot[(a, b)] += score_i * score_j

        neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
        min_sim = max(0.0, self.settings.cf_min_similarity)
        max_neighbors = max(1, self.settings.cf_max_neighbors)

        for (item_a, item_b), dot in pair_dot.items():
            denom = math.sqrt(item_norm_sq[item_a]) * math.sqrt(item_norm_sq[item_b])
            if denom <= 0:
                continue
            sim = dot / denom
            if sim < min_sim:
                continue
            neighbors[item_a].append((item_b, sim))
            neighbors[item_b].append((item_a, sim))

        for item_id in self.item_meta.keys():
            lst = neighbors.get(item_id, [])
            lst.sort(key=lambda x: x[1], reverse=True)
            neighbors[item_id] = lst[:max_neighbors]

        return dict(neighbors)

    def _meta(self) -> dict:
        """Metadata phục vụ debug/monitoring cho `/recommend/cache-status` và response."""
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

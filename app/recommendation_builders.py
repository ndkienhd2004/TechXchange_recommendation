from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

from app.config import Settings
from app.embedding_cache import EmbeddingCache
from app.gemini_embedding_client import GeminiEmbeddingClient
from app.repositories import RecommendationRepository
from app.vector_utils import add_scaled_dense, normalize_dense


def build_item_meta(products: list[dict]) -> dict[int, dict]:
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


def product_to_embedding_text(row: dict) -> str:
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


def build_item_embeddings(
    products: list[dict],
    settings: Settings,
    embedding_cache: EmbeddingCache,
    embedding_client: GeminiEmbeddingClient,
) -> dict[int, list[float]]:
    if not products:
        return {}

    texts: dict[int, str] = {}
    for row in products:
        pid = int(row["id"])
        texts[pid] = product_to_embedding_text(row)

    embedding_cache.load()
    embedding_cache.ensure_model(settings.gemini_embed_model)
    vectors: dict[int, list[float]] = {}
    to_embed: list[tuple[int, str, str]] = []

    for pid, text in texts.items():
        content_hash = embedding_cache.build_hash(text)
        cached = embedding_cache.get(pid, content_hash)
        if cached:
            vectors[pid] = normalize_dense(cached)
            continue
        to_embed.append((pid, text, content_hash))

    if to_embed:
        if settings.google_genai_use_vertexai:
            if not settings.google_cloud_project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex AI.")
            if not settings.google_cloud_location:
                raise RuntimeError("GOOGLE_CLOUD_LOCATION is required for Vertex AI.")
        elif not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when GOOGLE_GENAI_USE_VERTEXAI=false")

        embed_texts = [x[1] for x in to_embed]
        new_vectors = embedding_client.embed_texts(embed_texts)
        if len(new_vectors) != len(to_embed):
            raise RuntimeError(
                f"Embedding size mismatch: expected {len(to_embed)}, got {len(new_vectors)}"
            )
        for idx, (pid, _text, content_hash) in enumerate(to_embed):
            vec = normalize_dense(new_vectors[idx])
            embedding_cache.set(pid, content_hash, vec)
            vectors[pid] = vec

    embedding_cache.delete_missing(set(texts.keys()))
    embedding_cache.save(settings.gemini_embed_model)
    return vectors


def build_popularity_map(repo: RecommendationRepository, lookback_days: int) -> dict[int, float]:
    rows = repo.fetch_product_popularity(lookback_days)
    output: dict[int, float] = {}
    for row in rows:
        pid = int(row["product_id"])
        output[pid] = float(row["pop_score"] or 0.0)
    return output


def build_user_event_features(
    event_rows: list[dict],
    item_vectors: dict[int, list[float]],
    event_weights: dict[str, float],
) -> tuple[dict[int, dict], dict[int, dict[int, float]], dict[int, set[int]]]:
    now = dt.datetime.now(dt.timezone.utc)
    weighted: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    seen_products: dict[int, set[int]] = defaultdict(set)

    for row in event_rows:
        user_id = int(row["user_id"])
        product_id = int(row["product_id"])
        if product_id not in item_vectors:
            continue

        event_type = str(row["event_type"])
        event_count = int(row["event_count"] or 0)
        base_weight = event_weights.get(event_type, 0.0) * max(event_count, 1)

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
            vector = add_scaled_dense(vector, item_vectors[pid], scale)
        content_profiles[uid] = {
            "vector": normalize_dense(vector),
            "seen_products": seen_products.get(uid, set()),
        }

    user_item_scores = {uid: dict(items) for uid, items in weighted.items()}
    user_seen = {uid: set(items) for uid, items in seen_products.items()}
    return content_profiles, user_item_scores, user_seen


def build_cf_item_neighbors(
    user_item_scores: dict[int, dict[int, float]],
    item_ids: list[int],
    min_similarity: float,
    max_neighbors: int,
) -> dict[int, list[tuple[int, float]]]:
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
    min_sim = max(0.0, min_similarity)
    max_neighbor_count = max(1, max_neighbors)

    for (item_a, item_b), dot in pair_dot.items():
        denom = math.sqrt(item_norm_sq[item_a]) * math.sqrt(item_norm_sq[item_b])
        if denom <= 0:
            continue
        sim = dot / denom
        if sim < min_sim:
            continue
        neighbors[item_a].append((item_b, sim))
        neighbors[item_b].append((item_a, sim))

    for item_id in item_ids:
        lst = neighbors.get(item_id, [])
        lst.sort(key=lambda x: x[1], reverse=True)
        neighbors[item_id] = lst[:max_neighbor_count]

    return dict(neighbors)


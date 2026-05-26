from __future__ import annotations

from collections import defaultdict

from app.vector_utils import cosine_dense


def fallback_popular(
    item_meta: dict[int, dict],
    popularity: dict[int, float],
    limit: int,
) -> list[dict]:
    pop_max = max(popularity.values()) if popularity else 0.0
    rows: list[dict] = []
    for product_id, meta in item_meta.items():
        pop = popularity.get(product_id, 0.0)
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


def recommend_content_based_cached(
    user_id: int,
    limit: int,
    user_profiles: dict[int, dict],
    item_vectors: dict[int, list[float]],
    item_meta: dict[int, dict],
    popularity: dict[int, float],
) -> list[dict]:
    limit = max(1, min(limit, 100))
    profile = user_profiles.get(user_id)
    if not profile or not profile.get("vector"):
        return fallback_popular(item_meta=item_meta, popularity=popularity, limit=limit)

    user_vector = profile["vector"]
    seen_products = profile["seen_products"]
    pop_max = max(popularity.values()) if popularity else 0.0

    unseen_rows: list[dict] = []
    seen_rows: list[dict] = []
    for product_id, item_vector in item_vectors.items():
        content_score = cosine_dense(user_vector, item_vector)
        if content_score <= 0:
            continue
        pop = popularity.get(product_id, 0.0)
        pop_norm = (pop / pop_max) if pop_max > 0 else 0.0
        final_score = (content_score * 0.9) + (pop_norm * 0.1)
        meta = item_meta.get(product_id, {})
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


def recommend_collaborative_cached(
    user_id: int,
    limit: int,
    user_item_scores: dict[int, dict[int, float]],
    user_seen_products: dict[int, set[int]],
    cf_item_neighbors: dict[int, list[tuple[int, float]]],
    item_meta: dict[int, dict],
    popularity: dict[int, float],
) -> list[dict]:
    limit = max(1, min(limit, 100))
    user_scores = user_item_scores.get(user_id)
    if not user_scores:
        return fallback_popular(item_meta=item_meta, popularity=popularity, limit=limit)

    seen_products = user_seen_products.get(user_id, set())
    raw_scores: dict[int, float] = defaultdict(float)
    sim_sums: dict[int, float] = defaultdict(float)

    for item_id, interaction_score in user_scores.items():
        for neighbor_id, sim in cf_item_neighbors.get(item_id, []):
            raw_scores[neighbor_id] += sim * interaction_score
            sim_sums[neighbor_id] += abs(sim)

    if not raw_scores:
        return fallback_popular(item_meta=item_meta, popularity=popularity, limit=limit)

    collab_scores: dict[int, float] = {}
    for pid, val in raw_scores.items():
        denom = sim_sums.get(pid, 0.0)
        collab_scores[pid] = (val / denom) if denom > 0 else val

    max_collab = max(collab_scores.values()) if collab_scores else 0.0
    pop_max = max(popularity.values()) if popularity else 0.0

    unseen_rows: list[dict] = []
    seen_rows: list[dict] = []
    for pid, val in collab_scores.items():
        collab_norm = (val / max_collab) if max_collab > 0 else 0.0
        pop_norm = (popularity.get(pid, 0.0) / pop_max) if pop_max > 0 else 0.0
        final_score = (collab_norm * 0.95) + (pop_norm * 0.05)
        meta = item_meta.get(pid, {})
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


def similar_products_content(
    product_id: int,
    limit: int,
    item_vectors: dict[int, list[float]],
    item_meta: dict[int, dict],
) -> list[dict]:
    limit = max(1, min(limit, 100))
    base_vector = item_vectors.get(product_id)
    base_meta = item_meta.get(product_id)
    if not base_vector or not base_meta:
        raise ValueError(f"Product {product_id} not found or not active")

    rows: list[dict] = []
    for other_id, other_vec in item_vectors.items():
        if other_id == product_id:
            continue
        content_score = cosine_dense(base_vector, other_vec)
        if content_score <= 0:
            continue
        other_meta = item_meta.get(other_id, {})

        boost = 0.0
        if other_meta.get("category_id") == base_meta.get("category_id"):
            boost += 0.05
        if base_meta.get("brand_id") and (other_meta.get("brand_id") == base_meta.get("brand_id")):
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
    return rows[:limit]


def similar_products_collaborative(
    product_id: int,
    limit: int,
    cf_item_neighbors: dict[int, list[tuple[int, float]]],
    item_meta: dict[int, dict],
) -> list[dict]:
    limit = max(1, min(limit, 100))
    if product_id not in item_meta:
        raise ValueError(f"Product {product_id} not found or not active")

    neighbors = cf_item_neighbors.get(product_id, [])
    rows: list[dict] = []
    for other_id, sim in neighbors[:limit]:
        meta = item_meta.get(other_id, {})
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
    return rows


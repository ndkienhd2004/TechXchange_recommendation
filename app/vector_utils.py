from __future__ import annotations

import math


def normalize_dense(values: list[float]) -> list[float]:
    """Chuẩn hóa vector dense về unit length (L2 norm).

    Trả nguyên nếu vector rỗng hoặc norm <= 0.
    """
    if not values:
        return values
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [v / norm for v in values]


def cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity cho 2 vector đã normalize.

    Vì recommender luôn normalize embedding vector, phép tính ở đây chỉ cần dot product.
    Nếu vector khác chiều, chỉ dùng phần prefix chung `min(len(a), len(b))`.
    """
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    return sum(a[i] * b[i] for i in range(size))


def add_scaled_dense(base: list[float], other: list[float], scale: float) -> list[float]:
    """Cộng `base += other * scale` (in-place khi base đã có).

    Dùng để build user profile vector = weighted average của các item vectors.
    Nếu base rỗng, hàm trả vector mới (không mutate input).
    """
    if not other:
        return base
    if not base:
        return [v * scale for v in other]
    size = min(len(base), len(other))
    for i in range(size):
        base[i] += other[i] * scale
    return base

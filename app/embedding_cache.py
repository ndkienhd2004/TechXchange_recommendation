from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Một entry cache cho embedding của 1 product tại 1 thời điểm nội dung."""
    content_hash: str
    vector: list[float]
    updated_at: str


class EmbeddingCache:
    def __init__(self, path: str) -> None:
        """Cache embedding vectors theo `product_id` và `content_hash` (SHA-256 của text).

        Định dạng file JSON:
        - model: tên embedding model (đổi model sẽ invalid toàn bộ cache)
        - items: map product_id -> {content_hash, vector, updated_at}
        """
        self.path = path
        self._items: dict[str, CacheEntry] = {}
        self._model: str = ""
        self._loaded = False

    def load(self) -> None:
        """Load cache từ disk một lần (idempotent)."""
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.path):
            self._items = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._model = str(data.get("model") or "")
        items = data.get("items") or {}
        parsed: dict[str, CacheEntry] = {}
        for pid, row in items.items():
            parsed[str(pid)] = CacheEntry(
                content_hash=str(row.get("content_hash") or ""),
                vector=[float(v) for v in (row.get("vector") or [])],
                updated_at=str(row.get("updated_at") or ""),
            )
        self._items = parsed

    def save(self, model: str) -> None:
        """Ghi cache ra disk, tạo thư mục cha nếu cần."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "model": model,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "items": {
                pid: {
                    "content_hash": entry.content_hash,
                    "vector": entry.vector,
                    "updated_at": entry.updated_at,
                }
                for pid, entry in self._items.items()
            },
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        self._model = model

    def ensure_model(self, model: str) -> None:
        """Đảm bảo cache tương ứng đúng embedding model; khác model → reset cache."""
        self.load()
        if self._model and self._model != model:
            self._items = {}
        self._model = model

    @staticmethod
    def build_hash(text: str) -> str:
        """Tạo content hash ổn định để cache embedding theo nội dung."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, product_id: int, content_hash: str) -> list[float] | None:
        """Lấy vector nếu cache-hit theo (product_id, content_hash)."""
        self.load()
        row = self._items.get(str(product_id))
        if not row:
            return None
        if row.content_hash != content_hash:
            return None
        return row.vector

    def set(self, product_id: int, content_hash: str, vector: list[float]) -> None:
        """Set/overwrite vector cache cho product + content_hash hiện tại."""
        self.load()
        self._items[str(product_id)] = CacheEntry(
            content_hash=content_hash,
            vector=vector,
            updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    def delete_missing(self, keep_product_ids: set[int]) -> None:
        """Xóa cache entries cho product không còn trong tập active."""
        self.load()
        keys_to_keep = {str(pid) for pid in keep_product_ids}
        self._items = {k: v for k, v in self._items.items() if k in keys_to_keep}

    def stats(self) -> dict:
        """Trả thống kê nhẹ để expose qua `/recommend/cache-status`."""
        self.load()
        return {"entries": len(self._items), "model": self._model}

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _as_int(name: str, default: int) -> int:
    """Đọc biến môi trường dạng int; fallback về default nếu thiếu/lỗi parse."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    """Đọc biến môi trường dạng boolean theo các giá trị truthy phổ biến."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Cấu hình runtime đọc từ environment variables (có default an toàn).

    Nhóm cấu hình:
    - DB: kết nối Postgres
    - APP: uvicorn host/port/log/reload
    - Lookback/refresh: thời gian lấy dữ liệu events/popularity và TTL refresh
    - Embedding: Gemini embedding model + cache path
    - CF/Hybrid: tham số collaborative neighbors và trọng số hybrid
    """
    db_name: str = os.getenv("DB_NAME", "techxchange")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_pass: str = os.getenv("DB_PASS", "postgres")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = _as_int("DB_PORT", 5432)

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = _as_int("APP_PORT", 8010)
    log_level: str = os.getenv("LOG_LEVEL", "info")
    app_reload: bool = _as_bool("APP_RELOAD", False)

    event_lookback_days: int = _as_int("EVENT_LOOKBACK_DAYS", 60)
    popularity_lookback_days: int = _as_int("POPULARITY_LOOKBACK_DAYS", 30)
    refresh_interval_minutes: int = _as_int("REFRESH_INTERVAL_MINUTES", 15)

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    gemini_embed_dim: int = _as_int("GEMINI_EMBED_DIM", 768)
    gemini_batch_size: int = _as_int("GEMINI_BATCH_SIZE", 32)
    gemini_timeout_seconds: int = _as_int("GEMINI_TIMEOUT_SECONDS", 30)
    embedding_cache_path: str = os.getenv(
        "EMBEDDING_CACHE_PATH", ".cache/gemini_item_embeddings.json"
    )
    cf_min_similarity: float = float(os.getenv("CF_MIN_SIMILARITY", "0.05"))
    cf_max_neighbors: int = _as_int("CF_MAX_NEIGHBORS", 50)
    hybrid_content_weight: float = float(os.getenv("HYBRID_CONTENT_WEIGHT", "0.6"))


settings = Settings()

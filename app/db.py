from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from app.config import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
        """Quản lý pool kết nối Postgres (psycopg2 SimpleConnectionPool)."""
        self.settings = settings
        self.pool: SimpleConnectionPool | None = None

    def connect(self) -> None:
        """Khởi tạo pool nếu chưa tồn tại (idempotent)."""
        if self.pool is not None:
            return
        self.pool = SimpleConnectionPool(
            minconn=1,
            maxconn=8,
            dbname=self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_pass,
            host=self.settings.db_host,
            port=self.settings.db_port,
        )

    def close(self) -> None:
        """Đóng toàn bộ connection trong pool (dùng khi shutdown)."""
        if self.pool is not None:
            self.pool.closeall()
            self.pool = None

    @contextmanager
    def connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Lấy 1 connection từ pool và trả về lại pool sau khi dùng."""
        if self.pool is None:
            raise RuntimeError("Database pool is not initialized")
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def fetch_all(self, query: str, params: tuple | None = None) -> list[dict]:
        """Chạy query SELECT và trả list[dict] (RealDictCursor → dict thuần)."""
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]

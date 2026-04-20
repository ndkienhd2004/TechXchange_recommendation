from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.db import Database
from app.recommender import ContentBasedRecommender
from app.repositories import RecommendationRepository
from app.routes import build_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook.

    Khởi tạo các dependency dùng chung cho toàn app:
    - Tạo pool DB (`Database.connect`)
    - Tạo repository truy vấn dữ liệu recommendation
    - Tạo `ContentBasedRecommender` và build cache ban đầu (`refresh`)
    - Mount router `/recommend/*` với recommender đã sẵn sàng

    Khi shutdown: đóng pool DB.
    """
    db = Database(settings)
    db.connect()
    repo = RecommendationRepository(db)
    recommender = ContentBasedRecommender(repo, settings)
    recommender.refresh()

    app.state.db = db
    app.state.recommender = recommender
    app.include_router(build_router(recommender))
    yield
    db.close()


app = FastAPI(
    title="TechXchange Recommendation System",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Health check đơn giản, không phụ thuộc DB/cache."""
    return {"status": "ok"}


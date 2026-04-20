from fastapi import APIRouter, HTTPException, Query

from app.recommender import ContentBasedRecommender


def build_router(recommender: ContentBasedRecommender) -> APIRouter:
    """Tạo router `/recommend/*` và bind trực tiếp vào instance recommender.

    Lưu ý: `recommender` chứa state cache in-memory (vectors, profiles, CF neighbors).
    Các handler bên dưới chủ yếu gọi các public API của `ContentBasedRecommender`
    và wrap lỗi domain (ví dụ product không tồn tại) thành HTTP status phù hợp.
    """
    router = APIRouter(prefix="/recommend", tags=["recommend"])

    @router.get("/content-based/{user_id}")
    def recommend_content_based(
        user_id: int, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict:
        """Use case: recommend content-based cho user.

        - Nếu user có profile (từ events) → score theo cosine(user_profile, item_vector)
        - Nếu user cold-start → fallback theo popularity
        """
        return recommender.recommend_for_user(user_id=user_id, limit=limit)

    @router.get("/similar/{product_id}")
    def recommend_similar(
        product_id: int, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict:
        """Use case: item-to-item similarity theo content embedding.

        Trả 404 nếu `product_id` không active/không có embedding vector.
        """
        try:
            return recommender.similar_products(product_id=product_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/collaborative/{user_id}")
    def recommend_collaborative(
        user_id: int, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict:
        """Use case: collaborative filtering (item-item) cho user.

        - Dựa trên `user_product_events` để build implicit feedback score.
        - Nếu user không có lịch sử / neighbor trống → fallback popularity.
        """
        return recommender.recommend_collaborative_for_user(user_id=user_id, limit=limit)

    @router.get("/hybrid/{user_id}")
    def recommend_hybrid(
        user_id: int, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict:
        """Use case: hybrid content + collaborative.

        Score = alpha*content + (1-alpha)*collab, với alpha cấu hình qua env.
        Nếu user không có collaborative state → alpha tự ép về 1.0.
        """
        return recommender.recommend_hybrid_for_user(user_id=user_id, limit=limit)

    @router.get("/similar-collaborative/{product_id}")
    def recommend_similar_collaborative(
        product_id: int, limit: int = Query(default=10, ge=1, le=100)
    ) -> dict:
        """Use case: item-to-item similarity theo collaborative neighbors.

        Trả 404 nếu product không active/không có trong item_meta.
        """
        try:
            return recommender.similar_products_collaborative(product_id=product_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/refresh")
    def refresh_now() -> dict:
        """Use case: refresh thủ công (rebuild toàn bộ cache).

        Thao tác này chạy đồng bộ trong request thread; `ContentBasedRecommender`
        có lock để tránh refresh đồng thời.
        """
        recommender.refresh()
        return {"ok": True, "meta": recommender._meta()}

    @router.get("/cache-status")
    def cache_status() -> dict:
        """Use case: kiểm tra trạng thái cache/refresh.

        `ensure_fresh()` có thể trigger refresh nếu quá hạn.
        """
        recommender.ensure_fresh()
        return {"ok": True, "meta": recommender._meta()}

    return router

# TechXchange Recommendation System (Python)

Base service cho recommendation theo hướng **content-based filtering** dùng **Gemini Embedding 1**.

## 1) Cài đặt

```bash
cd "/Users/kien/Codes/TechXchange_recommendation_system"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Điền thông tin DB trong `.env` (trùng DB TechXchange backend hiện tại) + `GEMINI_API_KEY`.
`APP_RELOAD=false` mặc định để tránh spawn nhiều process khi chạy production.

## 2) Chạy service

```bash
python run.py
```

Hoặc:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## 3) API

- `GET /health`
- `GET /recommend/content-based/{user_id}?limit=10`
- `GET /recommend/collaborative/{user_id}?limit=10`
- `GET /recommend/hybrid/{user_id}?limit=10`
- `GET /recommend/similar/{product_id}?limit=10`
- `GET /recommend/similar-collaborative/{product_id}?limit=10`
- `POST /recommend/refresh`
- `GET /recommend/cache-status`

Ví dụ:

```bash
curl "http://localhost:8010/recommend/content-based/4?limit=5"
curl "http://localhost:8010/recommend/collaborative/4?limit=5"
curl "http://localhost:8010/recommend/hybrid/4?limit=5"
curl "http://localhost:8010/recommend/similar/5?limit=5"
curl "http://localhost:8010/recommend/similar-collaborative/5?limit=5"
```

## 4) Content-based base logic

- Item profile: embed text sản phẩm bằng `gemini-embedding-001`.
- User profile: weighted average embedding từ sản phẩm user đã tương tác (theo event weight + recency).
- Scoring: cosine similarity embedding + popularity boost.
- Cold start: fallback theo popularity gần đây.

## 5) Cơ chế tăng tốc

- DB chỉ bị query khi `refresh()`:
  - load active products
  - load popularity
  - load toàn bộ user events (lookback)
- Sau đó request runtime chỉ query trên cache RAM:
  - `item_vectors`
  - `user_profiles`
  - `popularity`
- Embedding sản phẩm được lưu local tại `EMBEDDING_CACHE_PATH` để không gọi Gemini lại nếu content không đổi.

## 6) Collaborative Filtering

- Dữ liệu collaborative lấy từ `user_product_events` (implicit feedback).
- Trọng số event:
  - `impression=1, view=2, click=3, add_to_cart=6, wishlist=7, purchase=12`
- Build `user_item_scores` theo recency decay (`exp(-age_days/30)`).
- Dựng item-item similarity bằng cosine trên không gian user interactions.
- `GET /recommend/collaborative/{user_id}`:
  - Dùng các item user đã tương tác làm seed.
  - Lan truyền điểm qua item neighbors (weighted by similarity).
  - Cộng thêm popularity boost nhẹ.

## 7) Hybrid

- `GET /recommend/hybrid/{user_id}` kết hợp:
  - `content score` (Gemini embedding)
  - `collaborative score` (item-item CF)
- Công thức:
  - `hybrid = alpha * content + (1-alpha) * collaborative`
  - `alpha` lấy từ `HYBRID_CONTENT_WEIGHT` (mặc định `0.6`)

## 8) Cấu trúc thư mục

```text
app/
  config.py
  db.py
  embedding_cache.py
  gemini_embedding_client.py
  repositories.py
  recommender.py
  routes.py
  vector_utils.py
  main.py
run.py
requirements.txt
```

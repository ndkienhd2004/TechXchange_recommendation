## Tổng quan hệ thống

Hệ thống là một API FastAPI cung cấp các endpoint recommendation dưới prefix `/recommend`.
Tại startup (lifespan), app:

- Khởi tạo `Database` (pool kết nối Postgres)
- Khởi tạo `RecommendationRepository` (chứa query lấy product/events)
- Khởi tạo `ContentBasedRecommender` (chứa toàn bộ state + embedding client/cache)
- Gọi `recommender.refresh()` để build cache ban đầu
- Mount router `/recommend/*` bằng `build_router(recommender)`

Các thành phần chính:

- **API layer**: `app/main.py`, `app/routes.py`
- **Recommender core**: `app/recommender.py`
- **DB access**: `app/db.py`, `app/repositories.py`
- **Embedding**: `app/gemini_embedding_client.py`, `app/embedding_cache.py`
- **Vector math**: `app/vector_utils.py`
- **Config**: `app/config.py` (đọc env + defaults)

---

## Use case 0: Khởi động service (startup/lifespan)

### Trigger

- Chạy `python run.py` hoặc chạy uvicorn trỏ vào `app.main:app`.

### Luồng chi tiết

1. `uvicorn.run("app.main:app", ...)` import module `app.main`.
2. FastAPI gọi `lifespan(app)` (async context manager).
3. `Database(settings)` tạo object pool manager.
4. `db.connect()` tạo `SimpleConnectionPool(min=1, max=8)`.
5. `RecommendationRepository(db)` tạo repo.
6. `ContentBasedRecommender(repo, settings)` tạo recommender:
   - Nạp config weights, refresh interval, CF params
   - Khởi tạo `EmbeddingCache(path)` và `GeminiEmbeddingClient(...)`
7. `recommender.refresh()` build toàn bộ state cache:
   - `repo.fetch_active_products()` lấy danh sách product active + join catalog/brand/category
   - `_build_item_meta(products)` tạo map `product_id -> meta`
   - `_build_item_embeddings(products)` tạo embedding vector cho từng product:
     - Build text bằng `_product_to_embedding_text(row)`
     - Load cache file `.cache/gemini_item_embeddings.json` (mặc định) nếu tồn tại
     - Nếu cache-hit (product_id + content_hash khớp) thì dùng vector cached
     - Nếu cache-miss:
       - Nếu `GEMINI_API_KEY` rỗng: **raise RuntimeError** (startup fail)
       - Gọi Gemini batch embed → normalize vector → lưu cache
     - Xóa cache entries cho product không còn active
     - Save cache ra disk
   - `_build_popularity_map()` tính pop_score theo lookback days
   - `repo.fetch_user_events_all(event_lookback_days)` lấy events theo user-product-event_type
   - `_build_user_event_features(events)`:
     - Tính score = EVENT_WEIGHT(event_type) * count * recency_decay
     - Lưu `user_item_scores` (phục vụ CF) + `seen_products`
     - Tạo `user_profiles[user_id]["vector"]` là weighted average embedding các product user đã tương tác
   - `_build_cf_item_neighbors(user_item_scores)`:
     - Tính cosine similarity item-item theo vector tương tác (implicit feedback)
     - Lọc theo `CF_MIN_SIMILARITY`, cắt top `CF_MAX_NEIGHBORS`
   - Set `last_refreshed_at = now(UTC)`
8. Gắn `app.state.db`, `app.state.recommender` và include router.
9. Service sẵn sàng nhận request.
10. Khi shutdown: `db.close()` đóng pool.

### Failure modes quan trọng

- **Không có GEMINI key và cache rỗng**: startup sẽ fail vì không build được embedding.
- **DB không connect được**: fail tại `db.connect()` hoặc query trong `refresh()`.
- **Cache file lỗi JSON**: fail khi `EmbeddingCache.load()`.

---

## Use case 1: Health check

### Endpoint

- `GET /health`

### Luồng

- Trả `{"status": "ok"}`. Không phụ thuộc DB hay cache.

---

## Use case 2: Recommend content-based cho user

### Endpoint

- `GET /recommend/content-based/{user_id}?limit=10`

### Luồng chi tiết

1. Router gọi `recommender.recommend_for_user(user_id, limit)`.
2. `ensure_fresh()`:
   - Nếu chưa refresh bao giờ → gọi `refresh()`.
   - Nếu quá `REFRESH_INTERVAL_MINUTES` → gọi `refresh()`.
3. `_recommend_content_based_cached(user_id, limit)`:
   - Nếu user không có `user_profiles[user_id]["vector"]` → fallback popularity.
   - Với mỗi product active chưa “seen”:
     - `content_score = cosine_dense(user_vector, item_vector)` (dot product vì vector đã normalize)
     - `pop_norm` = popularity normalized
     - `final_score = content_score*0.9 + pop_norm*0.1`
   - Sort desc, cắt `limit`.
4. `strategy`:
   - Nếu có item và `content_score > 0` → `content_based_gemini_embedding`
   - else → `content_based_fallback_popular`
5. Response:
   - `{"strategy", "user_id", "items", "meta"}`

### Output item schema (hiện tại)

- `product_id`, `name`, `price`, `category_id`, `brand_id`
- `score` (final), `content_score`, `popularity_boost`

### Edge cases

- **User mới (cold-start)**: không có profile → trả top popular.
- **Không có popularity map**: pop_max=0 → popularity_boost = 0.

---

## Use case 3: Recommend collaborative filtering cho user

### Endpoint

- `GET /recommend/collaborative/{user_id}?limit=10`

### Luồng chi tiết

1. Router gọi `recommender.recommend_collaborative_for_user(user_id, limit)`.
2. `ensure_fresh()` tương tự.
3. `_recommend_collaborative_cached(user_id, limit)`:
   - Nếu `user_item_scores[user_id]` không tồn tại → fallback popularity.
   - Với mỗi item user đã tương tác:
     - Lấy neighbors item-item từ `cf_item_neighbors[item_id]`
     - Accumulate:
       - `raw_scores[neighbor] += sim * interaction_score`
       - `sim_sums[neighbor] += abs(sim)`
   - Chuẩn hóa `collab_scores = raw / sim_sums`
   - Normalize theo max để có `collab_norm`
   - Blend nhẹ popularity: `final_score = collab_norm*0.95 + pop_norm*0.05`
   - Sort desc, cắt `limit`
4. `strategy`:
   - Nếu có item và `collab_score > 0` → `collaborative_filtering_item_item`
   - else → `collaborative_fallback_popular`

### Output item schema

- `product_id`, `name`, `price`, `category_id`, `brand_id`
- `score`, `collab_score`, `popularity_boost`

### Edge cases

- **User có events nhưng neighbors trống** (do min similarity quá cao, data sparse) → fallback popularity.

---

## Use case 4: Recommend hybrid (content + collaborative) cho user

### Endpoint

- `GET /recommend/hybrid/{user_id}?limit=10`

### Luồng chi tiết

1. `ensure_fresh()`.
2. Lấy candidate list “rộng”:
   - `content_rows = _recommend_content_based_cached(user_id, max(limit*3, 30))`
   - `collab_rows = _recommend_collaborative_cached(user_id, max(limit*3, 30))`
3. Xác định hệ số `alpha = HYBRID_CONTENT_WEIGHT` clamp 0..1.
   - Nếu user không có data collaborative (`user_id not in user_item_scores`) → alpha=1.0.
4. Tạo union product IDs từ 2 list, tính:
   - `hybrid_score = alpha*content_score + (1-alpha)*collab_score`
5. Sort và trả top `limit`.

### Output item schema

- `score` (hybrid)
- `content_component`, `collaborative_component`
- + meta sản phẩm cơ bản

---

## Use case 5: Tìm sản phẩm tương tự theo content embedding (item-to-item)

### Endpoint

- `GET /recommend/similar/{product_id}?limit=10`

### Luồng chi tiết

1. `ensure_fresh()`.
2. Lấy `base_vector` và `base_meta`; nếu thiếu → raise `ValueError`.
3. Duyệt toàn bộ `item_vectors`:
   - `content_score = cosine_dense(base_vector, other_vec)`
   - Boost heuristic:
     - +0.05 nếu cùng category
     - +0.03 nếu cùng brand (và base có brand_id)
   - `final_score = content_score + boost`
4. Sort, trả top `limit`.
5. Router bắt `ValueError` và trả `HTTP 404`.

### Output item schema

- `score` (final)
- `content_score` (raw similarity)
- + meta sản phẩm cơ bản

---

## Use case 6: Tìm sản phẩm tương tự theo collaborative (item-to-item neighbors)

### Endpoint

- `GET /recommend/similar-collaborative/{product_id}?limit=10`

### Luồng chi tiết

1. `ensure_fresh()`.
2. Nếu product không active (không có trong `item_meta`) → raise `ValueError` → router trả 404.
3. Lấy `neighbors = cf_item_neighbors.get(product_id, [])`.
4. Map neighbors thành rows (score = sim), cắt `limit`.

---

## Use case 7: Refresh thủ công

### Endpoint

- `POST /recommend/refresh`

### Luồng

- Gọi `recommender.refresh()` trong request thread.
- Trả `{ok: true, meta: recommender._meta()}`.

### Lưu ý vận hành

- `refresh()` có lock `_refresh_lock` để tránh refresh đồng thời.
- Trong lúc refresh, các request khác vẫn vào được nhưng sẽ block nếu gọi refresh/ensure_fresh cùng lock (do refresh chạy trong cùng process).

---

## Use case 8: Kiểm tra trạng thái cache/refresh

### Endpoint

- `GET /recommend/cache-status`

### Luồng

- Gọi `recommender.ensure_fresh()` (có thể trigger refresh).
- Trả `{ok: true, meta: recommender._meta()}`.

---

## Data contracts & tham số cấu hình chính (env)

- **DB**: `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`
- **APP**: `APP_HOST`, `APP_PORT`, `APP_RELOAD`, `LOG_LEVEL`
- **Lookback**: `EVENT_LOOKBACK_DAYS`, `POPULARITY_LOOKBACK_DAYS`
- **Refresh**: `REFRESH_INTERVAL_MINUTES`
- **Embedding**:
  - `GEMINI_API_KEY`
  - `GEMINI_EMBED_MODEL` (mặc định `gemini-embedding-001`)
  - `GEMINI_EMBED_DIM` (mặc định 768)
  - `GEMINI_BATCH_SIZE`
  - `GEMINI_TIMEOUT_SECONDS`
  - `EMBEDDING_CACHE_PATH` (mặc định `.cache/gemini_item_embeddings.json`)
- **Collaborative**:
  - `CF_MIN_SIMILARITY` (mặc định 0.05)
  - `CF_MAX_NEIGHBORS` (mặc định 50)
- **Hybrid**:
  - `HYBRID_CONTENT_WEIGHT` (mặc định 0.6)


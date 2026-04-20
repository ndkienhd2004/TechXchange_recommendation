from app.db import Database


class RecommendationRepository:
    def __init__(self, db: Database) -> None:
        """Data access layer cho recommendation: products + user events aggregates."""
        self.db = db

    def fetch_active_products(self) -> list[dict]:
        """Lấy các product 'active' còn hàng, join thêm catalog/brand/category metadata."""
        query = """
            SELECT
                p.id,
                p.category_id,
                p.brand_id,
                p.catalog_id,
                p.name,
                p.description,
                p.price::double precision AS price,
                p.quality,
                p.condition_percent::double precision AS condition_percent,
                p.rating::double precision AS rating,
                p.created_at,
                p.quantity,
                pc.name AS catalog_name,
                pc.description AS catalog_description,
                pc.specs::text AS catalog_specs,
                b.name AS brand_name,
                c.name AS category_name
            FROM products p
            LEFT JOIN product_catalog pc ON pc.id = p.catalog_id
            LEFT JOIN brand b ON b.id = p.brand_id
            LEFT JOIN product_categories c ON c.id = p.category_id
            WHERE p.status = 'active'
              AND COALESCE(p.quantity, 0) > 0
        """
        return self.db.fetch_all(query)

    def fetch_user_events_all(self, lookback_days: int) -> list[dict]:
        """Aggregate implicit events theo (user_id, product_id, event_type) trong lookback window."""
        query = """
            SELECT
                user_id,
                product_id,
                event_type::text AS event_type,
                COUNT(*)::int AS event_count,
                MAX(created_at) AS last_event_at
            FROM user_product_events
            WHERE created_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY user_id, product_id, event_type
        """
        return self.db.fetch_all(query, (lookback_days,))

    def fetch_product_popularity(self, lookback_days: int) -> list[dict]:
        """Tính popularity score theo product dựa trên weighted sum event_type trong lookback window."""
        query = """
            SELECT
                product_id,
                SUM(
                    CASE event_type::text
                        WHEN 'impression' THEN 1
                        WHEN 'view' THEN 2
                        WHEN 'click' THEN 3
                        WHEN 'add_to_cart' THEN 6
                        WHEN 'wishlist' THEN 7
                        WHEN 'purchase' THEN 12
                        ELSE 0
                    END
                )::double precision AS pop_score
            FROM user_product_events
            WHERE created_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY product_id
        """
        return self.db.fetch_all(query, (lookback_days,))

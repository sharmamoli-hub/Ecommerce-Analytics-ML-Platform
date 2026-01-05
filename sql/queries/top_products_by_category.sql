
SELECT 
    category,
    product_name,
    total_quantity,
    total_profit,
    rank
FROM (
    SELECT 
        p.category,
        p.product_name,
        SUM(oi.quantity) as total_quantity,
        ROUND(SUM(oi.profit), 2) as total_profit,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.profit) DESC) as rank
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.category, p.product_name
)
WHERE rank <= 3
ORDER BY category, rank

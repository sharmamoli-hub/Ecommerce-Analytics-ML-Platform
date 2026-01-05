
WITH country_sales AS (
    SELECT 
        c.country,
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT c.customer_id) as total_customers,
        SUM(oi.quantity) as total_quantity,
        ROUND(SUM(oi.profit), 2) as total_profit
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.country
),
country_ranks AS (
    SELECT 
        *,
        ROUND(total_profit * 100.0 / SUM(total_profit) OVER (), 2) as profit_percentage
    FROM country_sales
)
SELECT 
    country,
    total_orders,
    total_customers,
    total_quantity,
    total_profit,
    profit_percentage
FROM country_ranks
ORDER BY total_profit DESC
LIMIT 10

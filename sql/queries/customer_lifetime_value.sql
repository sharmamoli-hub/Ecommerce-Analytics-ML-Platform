
SELECT 
    c.customer_id,
    c.customer_name,
    c.country,
    c.customer_segment,
    COUNT(DISTINCT o.order_id) as total_orders,
    ROUND(julianday(MAX(o.order_date)) - julianday(MIN(o.order_date))) as customer_lifespan_days,
    ROUND(SUM(oi.profit), 2) as lifetime_value,
    ROUND(SUM(oi.profit) / COUNT(DISTINCT o.order_id), 2) as avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id, c.customer_name, c.country, c.customer_segment
HAVING total_orders >= 3
ORDER BY lifetime_value DESC
LIMIT 20

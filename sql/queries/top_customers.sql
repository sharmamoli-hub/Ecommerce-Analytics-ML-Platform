
SELECT 
    c.customer_id,
    c.customer_name,
    c.country,
    c.customer_segment,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.quantity) as total_items,
    ROUND(SUM(oi.profit), 2) as total_profit
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id, c.customer_name, c.country, c.customer_segment
ORDER BY total_profit DESC
LIMIT 10

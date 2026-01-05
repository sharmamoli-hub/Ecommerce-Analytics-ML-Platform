
SELECT 
    strftime('%Y-%m', o.order_date) as month,
    COUNT(DISTINCT o.order_id) as total_orders,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(oi.quantity) as total_items_sold,
    ROUND(SUM(oi.profit), 2) as total_profit
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY month
ORDER BY month

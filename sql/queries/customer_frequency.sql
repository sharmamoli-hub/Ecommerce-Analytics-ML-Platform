
SELECT 
    c.customer_segment,
    COUNT(DISTINCT c.customer_id) as num_customers,
    ROUND(AVG(order_count), 2) as avg_orders_per_customer,
    ROUND(AVG(total_profit), 2) as avg_profit_per_customer
FROM customers c
JOIN (
    SELECT 
        customer_id,
        COUNT(order_id) as order_count,
        SUM(profit) as total_profit
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY customer_id
) customer_stats ON c.customer_id = customer_stats.customer_id
GROUP BY c.customer_segment
ORDER BY avg_profit_per_customer DESC

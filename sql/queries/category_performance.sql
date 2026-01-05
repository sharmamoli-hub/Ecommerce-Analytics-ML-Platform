
SELECT 
    p.category,
    COUNT(DISTINCT p.product_id) as num_products,
    SUM(oi.quantity) as total_units_sold,
    ROUND(AVG(p.unit_price), 2) as avg_unit_price,
    ROUND(SUM(oi.profit), 2) as total_profit,
    ROUND(AVG(oi.profit), 2) as avg_profit_per_item
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.category
ORDER BY total_profit DESC

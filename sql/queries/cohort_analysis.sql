
WITH first_purchase AS (
    SELECT 
        customer_id,
        MIN(order_date) as first_order_date,
        strftime('%Y-%m', MIN(order_date)) as cohort_month
    FROM orders
    GROUP BY customer_id
),
customer_orders AS (
    SELECT 
        o.customer_id,
        fp.cohort_month,
        strftime('%Y-%m', o.order_date) as order_month,
        CAST((julianday(o.order_date) - julianday(fp.first_order_date)) / 30 AS INTEGER) as months_since_first
    FROM orders o
    JOIN first_purchase fp ON o.customer_id = fp.customer_id
)
SELECT 
    cohort_month,
    COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_id END) as month_0,
    COUNT(DISTINCT CASE WHEN months_since_first = 1 THEN customer_id END) as month_1,
    COUNT(DISTINCT CASE WHEN months_since_first = 2 THEN customer_id END) as month_2,
    COUNT(DISTINCT CASE WHEN months_since_first = 3 THEN customer_id END) as month_3
FROM customer_orders
GROUP BY cohort_month
ORDER BY cohort_month
LIMIT 6

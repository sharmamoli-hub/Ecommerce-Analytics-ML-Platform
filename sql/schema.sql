-- ============================================================
-- E-Commerce Database Schema
-- Created for Analytics Platform Project
-- ============================================================

-- Drop tables if they exist (for clean restart)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- ============================================================
-- CUSTOMERS TABLE
-- ============================================================
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    email TEXT,
    region TEXT,
    country TEXT,
    signup_date DATE,
    customer_segment TEXT
);

-- ============================================================
-- PRODUCTS TABLE
-- ============================================================
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    subcategory TEXT,
    unit_price REAL,
    cost_price REAL
);

-- ============================================================
-- ORDERS TABLE
-- ============================================================
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date DATE,
    ship_date DATE,
    ship_mode TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ============================================================
-- ORDER ITEMS TABLE (Junction table)
-- ============================================================
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    discount REAL,
    profit REAL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- ============================================================
-- CREATE INDEXES FOR PERFORMANCE
-- ============================================================
CREATE INDEX idx_customer_id ON orders(customer_id);
CREATE INDEX idx_order_date ON orders(order_date);
CREATE INDEX idx_product_id ON order_items(product_id);
CREATE INDEX idx_order_id ON order_items(order_id);

-- ============================================================
-- SCHEMA CREATED SUCCESSFULLY
-- ============================================================
"""
Database Setup Script
=====================
This script creates SQLite database and loads initial data
Author: Your Name
Date: December 2024
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

def create_database():
    """
    Create SQLite database and execute schema
    """
    print("=" * 60)
    print("STEP 1: SETTING UP DATABASE")
    print("=" * 60)
    
    # Database path
    db_path = 'data/ecommerce.db'
    
    # Remove old database if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✓ Removed old database")
    
    # Connect to SQLite (creates new database)
    conn = sqlite3.connect(db_path)
    print(f"✓ Created database at: {db_path}")
    
    # Read and execute schema
    try:
        with open('sql/schema.sql', 'r') as f:
            schema = f.read()
        
        conn.executescript(schema)
        print("✓ Database schema created successfully")
        print("✓ Tables created: customers, products, orders, order_items")
        print("✓ Indexes created for performance optimization")
        
    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        return None
    
    conn.commit()
    conn.close()
    
    print("\n✅ Database setup complete!")
    print("=" * 60)
    
    return db_path

def check_csv_file(csv_path):
    """
    Check if CSV file exists and show basic info
    """
    print("\n" + "=" * 60)
    print("STEP 2: CHECKING CSV FILE")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        print("\nPlease make sure:")
        print("1. You downloaded the dataset from Kaggle")
        print("2. You placed it in: data/raw/")
        print("3. You renamed it to: ecommerce_data.csv")
        return None
    
    print(f"✓ CSV file found at: {csv_path}")
    
    # Read CSV
    try:
        print("Loading CSV file...")
        df = pd.read_csv(csv_path, encoding='latin-1')
        print(f"✓ Successfully loaded {len(df):,} rows")
        print(f"✓ Found {len(df.columns)} columns")
        
        print("\n📊 DATASET PREVIEW:")
        print("-" * 60)
        print(df.head())
        
        print("\n📋 COLUMN INFORMATION:")
        print("-" * 60)
        print(df.info())
        
        print("\n📈 BASIC STATISTICS:")
        print("-" * 60)
        print(df.describe())
        
        print("\n✅ CSV file check complete!")
        print("=" * 60)
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def main():
    """
    Main execution function
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "E-COMMERCE DATABASE SETUP" + " " * 23 + "║")
    print("║" + " " * 15 + "Phase 1 - Project Foundation" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Step 1: Create database
    db_path = create_database()
    
    if db_path is None:
        print("\n❌ Database setup failed!")
        return
    
    # Step 2: Check CSV file
    csv_path = 'data/raw/ecommerce_data.csv'
    df = check_csv_file(csv_path)
    
    if df is None:
        print("\n⚠️  CSV file not ready yet")
        print("Database structure is created, but no data loaded.")
        print("\nNext steps:")
        print("1. Download dataset from Kaggle")
        print("2. Place it in data/raw/ folder")
        print("3. Run this script again")
    else:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! DATABASE READY WITH DATA")
        print("=" * 60)
        print(f"\n📊 Dataset Summary:")
        print(f"   - Total Records: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Date Range: {df['InvoiceDate'].min() if 'InvoiceDate' in df.columns else 'N/A'}")
        print(f"              to {df['InvoiceDate'].max() if 'InvoiceDate' in df.columns else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("NEXT STEP: Create Jupyter notebooks for data exploration")
    print("=" * 60)
    print("\n")

if __name__ == "__main__":
    main()
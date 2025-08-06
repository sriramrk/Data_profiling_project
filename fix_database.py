#!/usr/bin/env python3
"""
Database Migration and Repair Script
Fixes common database issues including the missing data_source column
"""

import sqlite3
import os
import sys

def check_and_fix_database(db_path="data/app.db"):
    """Check and fix common database issues."""
    
    print("🔧 Database Migration and Repair Tool")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("   The database will be created when you run the app.")
        return True
    
    print(f"🔍 Checking database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check file_metadata table structure
        print("\n📋 Checking file_metadata table...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata'")
        
        if cursor.fetchone():
            # Table exists, check columns
            cursor.execute("PRAGMA table_info(file_metadata)")
            columns = [column[1] for column in cursor.fetchall()]
            print(f"   Current columns: {', '.join(columns)}")
            
            if 'data_source' not in columns:
                print("   ⚠️  Missing 'data_source' column - fixing...")
                cursor.execute("ALTER TABLE file_metadata ADD COLUMN data_source TEXT DEFAULT 'uploaded'")
                print("   ✅ Added 'data_source' column")
            else:
                print("   ✅ 'data_source' column exists")
        else:
            print("   ℹ️  file_metadata table doesn't exist - will be created by app")
        
        # Check mimic_metadata table
        print("\n🏥 Checking mimic_metadata table...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='mimic_metadata'")
        
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM mimic_metadata")
            count = cursor.fetchone()[0]
            print(f"   ✅ mimic_metadata table exists with {count} entries")
        else:
            print("   ℹ️  mimic_metadata table doesn't exist - will be created by app")
        
        # Check query_history table
        print("\n📜 Checking query_history table...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_history'")
        
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM query_history")
            count = cursor.fetchone()[0]
            print(f"   ✅ query_history table exists with {count} entries")
        else:
            print("   ℹ️  query_history table doesn't exist - will be created by app")
        
        # List all data tables
        print("\n📊 Available data tables:")
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT IN ('file_metadata', 'mimic_metadata', 'query_history', 'sqlite_sequence')
        """)
        
        data_tables = cursor.fetchall()
        if data_tables:
            for table in data_tables:
                cursor.execute(f"SELECT COUNT(*) FROM `{table[0]}`")
                count = cursor.fetchone()[0]
                print(f"   📋 {table[0]}: {count:,} rows")
        else:
            print("   ℹ️  No data tables found")
        
        conn.commit()
        conn.close()
        
        print("\n✅ Database check and repair completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        return False

def backup_database(db_path="data/app.db"):
    """Create a backup of the database."""
    if not os.path.exists(db_path):
        print("No database to backup")
        return None
    
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

def clear_all_data(db_path="data/app.db"):
    """Clear all data tables but keep structure."""
    if not os.path.exists(db_path):
        print("No database found")
        return
    
    response = input("⚠️  This will delete ALL data. Type 'YES' to confirm: ")
    if response != 'YES':
        print("Operation cancelled")
        return
    
    try:
        # Create backup first
        backup_path = backup_database(db_path)
        if not backup_path:
            print("❌ Could not create backup - operation cancelled")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all data tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT IN ('file_metadata', 'mimic_metadata', 'query_history', 'sqlite_sequence')
        """)
        
        data_tables = cursor.fetchall()
        
        # Drop data tables
        for table in data_tables:
            cursor.execute(f"DROP TABLE IF EXISTS `{table[0]}`")
            print(f"   Dropped table: {table[0]}")
        
        # Clear metadata
        cursor.execute("DELETE FROM file_metadata")
        cursor.execute("DELETE FROM mimic_metadata")
        cursor.execute("DELETE FROM query_history")
        
        conn.commit()
        conn.close()
        
        print("✅ All data cleared successfully")
        
    except Exception as e:
        print(f"❌ Error clearing data: {e}")

def main():
    """Main function with menu options."""
    print("🔧 Data Profiling App - Database Repair Tool")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            check_and_fix_database()
        elif sys.argv[1] == '--backup':
            backup_database()
        elif sys.argv[1] == '--clear':
            clear_all_data()
        else:
            print("Usage: python fix_database.py [--check|--backup|--clear]")
        return
    
    while True:
        print("\nSelect an option:")
        print("1. 🔍 Check and fix database issues")
        print("2. 💾 Create database backup")
        print("3. 🗑️  Clear all data (keeps structure)")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            check_and_fix_database()
        elif choice == '2':
            backup_database()
        elif choice == '3':
            clear_all_data()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
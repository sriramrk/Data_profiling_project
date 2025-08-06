"""
Enhanced Database Manager - Fixed Version with Proper Schema Migration
"""

import sqlite3
import pandas as pd
import os
import glob
from typing import List, Dict, Any, Tuple, Optional
import re

class DatabaseManager:
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path
        self.mimic_data_path = "data/mimic_iii"  # MIMIC-III CSV files location
        self.ensure_data_directory()
        self.init_database()
        self.migrate_database()  # NEW: Add migration step
    
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.mimic_data_path, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create file_metadata table with all required columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                table_name TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER,
                column_count INTEGER,
                data_source TEXT DEFAULT 'uploaded'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                question TEXT NOT NULL,
                sql_query TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mimic_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL UNIQUE,
                description TEXT,
                row_count INTEGER,
                column_count INTEGER,
                load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size_mb REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def migrate_database(self):
        """Migrate existing database to add missing columns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if data_source column exists in file_metadata
            cursor.execute("PRAGMA table_info(file_metadata)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'data_source' not in columns:
                print("Migrating database: Adding data_source column...")
                cursor.execute('ALTER TABLE file_metadata ADD COLUMN data_source TEXT DEFAULT "uploaded"')
                print("Database migration completed successfully")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database migration error: {e}")
            # If migration fails, we might need to recreate the table
            self._recreate_file_metadata_table()
    
    def _recreate_file_metadata_table(self):
        """Recreate file_metadata table with proper schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Backup existing data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata'")
            if cursor.fetchone():
                try:
                    cursor.execute("SELECT * FROM file_metadata")
                    existing_data = cursor.fetchall()
                    cursor.execute("PRAGMA table_info(file_metadata)")
                    old_columns = [column[1] for column in cursor.fetchall()]
                    
                    # Drop old table
                    cursor.execute("DROP TABLE file_metadata")
                    
                    # Create new table
                    cursor.execute('''
                        CREATE TABLE file_metadata (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT NOT NULL,
                            table_name TEXT NOT NULL,
                            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            row_count INTEGER,
                            column_count INTEGER,
                            data_source TEXT DEFAULT 'uploaded'
                        )
                    ''')
                    
                    # Restore data with default data_source
                    for row in existing_data:
                        if len(row) == 6:  # Old format without data_source
                            cursor.execute('''
                                INSERT INTO file_metadata 
                                (id, filename, table_name, upload_date, row_count, column_count, data_source)
                                VALUES (?, ?, ?, ?, ?, ?, 'uploaded')
                            ''', row)
                        else:  # New format with data_source
                            cursor.execute('''
                                INSERT INTO file_metadata 
                                (id, filename, table_name, upload_date, row_count, column_count, data_source)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', row)
                    
                    print("File metadata table recreated successfully")
                    
                except Exception as e:
                    print(f"Error recreating table: {e}")
                    # Create empty table if all else fails
                    cursor.execute("DROP TABLE IF EXISTS file_metadata")
                    cursor.execute('''
                        CREATE TABLE file_metadata (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT NOT NULL,
                            table_name TEXT NOT NULL,
                            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            row_count INTEGER,
                            column_count INTEGER,
                            data_source TEXT DEFAULT 'uploaded'
                        )
                    ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error in table recreation: {e}")
    
    def upload_csv_data(self, df: pd.DataFrame, filename: str, table_name: str) -> Dict[str, Any]:
        """Upload CSV data to database and store metadata."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store the data
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Save metadata with data_source tracking
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (filename, table_name, row_count, column_count, data_source)
                VALUES (?, ?, ?, ?, 'uploaded')
            ''', (filename, table_name, len(df), len(df.columns)))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': f'Successfully uploaded {filename} with {len(df)} rows and {len(df.columns)} columns!',
                'metadata': {
                    'filename': filename,
                    'table_name': table_name,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'data_source': 'uploaded'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error uploading data: {str(e)}',
                'metadata': None
            }
    
    def load_mimic_data(self) -> Dict[str, Any]:
        """Load MIMIC-III data from CSV files if available."""
        try:
            # Look for CSV files
            csv_files = glob.glob(os.path.join(self.mimic_data_path, "*.csv"))
            
            if not csv_files:
                return {
                    'success': False,
                    'message': f'No CSV files found in {self.mimic_data_path}. Please place MIMIC-III CSV files there first.',
                    'tables_loaded': []
                }
            
            # Check if already loaded (but allow re-loading)
            existing_tables = self.get_mimic_tables()
            if existing_tables:
                # If data exists, give option to reload or use existing
                return {
                    'success': True,
                    'message': f'MIMIC-III data already loaded with {len(existing_tables)} tables',
                    'tables_loaded': existing_tables,
                    'already_loaded': True
                }
            
            # Table descriptions for MIMIC-III
            table_descriptions = {
                'admissions': 'Hospital admissions data',
                'patients': 'Patient demographics and information', 
                'icustays': 'ICU stay records',
                'chartevents': 'Chart events and vital signs',
                'labevents': 'Laboratory test results',
                'prescriptions': 'Medication prescriptions',
                'diagnoses_icd': 'ICD diagnosis codes',
                'procedures_icd': 'ICD procedure codes',
                'noteevents': 'Clinical notes and reports',
                'inputevents_cv': 'Input events (CareVue)',
                'inputevents_mv': 'Input events (MetaVision)',
                'outputevents': 'Output events',
                'microbiologyevents': 'Microbiology test results',
                'datetimeevents': 'Date/time events',
                'cptevents': 'CPT events',
                'callout': 'Patient callout records',
                'caregivers': 'Caregiver information',
                'drgcodes': 'DRG codes',
                'services': 'Hospital services',
                'transfers': 'Patient transfers',
                'd_cpt': 'CPT code dictionary',
                'd_icd_diagnoses': 'ICD diagnosis dictionary',
                'd_icd_procedures': 'ICD procedure dictionary', 
                'd_items': 'Items dictionary',
                'd_labitems': 'Lab items dictionary'
            }
            
            conn = sqlite3.connect(self.db_path)
            loaded_tables = []
            
            print(f"Loading MIMIC-III data from {len(csv_files)} files...")
            
            for csv_file in csv_files:
                try:
                    filename = os.path.basename(csv_file)
                    table_name = filename.replace('.csv', '').lower()
                    
                    print(f"Processing {filename}...")
                    
                    # Read CSV with error handling
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False)
                    
                    # Clean column names
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                    
                    # Sample large files for demo purposes
                    original_rows = len(df)
                    if len(df) > 100000:
                        df = df.sample(n=100000, random_state=42)
                        print(f"  Sampled {filename} from {original_rows:,} to {len(df):,} rows for demo")
                    
                    # Load to database
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    
                    # Save metadata
                    file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                    description = table_descriptions.get(table_name, f'MIMIC-III {table_name} table')
                    
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO mimic_metadata 
                        (table_name, description, row_count, column_count, file_size_mb)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (table_name, description, len(df), len(df.columns), file_size_mb))
                    
                    # Also add to file_metadata for consistency
                    cursor.execute('''
                        INSERT OR REPLACE INTO file_metadata 
                        (filename, table_name, row_count, column_count, data_source)
                        VALUES (?, ?, ?, ?, 'mimic')
                    ''', (filename, table_name, len(df), len(df.columns)))
                    
                    loaded_tables.append({
                        'table_name': table_name,
                        'description': description,
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'original_rows': original_rows
                    })
                    
                    print(f"  ✓ Loaded {table_name}: {len(df):,} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {str(e)}")
                    continue
            
            conn.commit()
            conn.close()
            
            if loaded_tables:
                return {
                    'success': True,
                    'message': f'Successfully loaded {len(loaded_tables)} MIMIC-III tables!',
                    'tables_loaded': loaded_tables
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to load any MIMIC-III tables. Check the CSV files and try again.',
                    'tables_loaded': []
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error loading MIMIC-III data: {str(e)}',
                'tables_loaded': []
            }
    
    def is_mimic_data_loaded(self) -> bool:
        """Check if MIMIC-III data is already loaded."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM mimic_metadata")
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except:
            return False
    
    def get_mimic_tables(self) -> List[Dict[str, Any]]:
        """Get list of available MIMIC-III tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT table_name, description, row_count, column_count 
                FROM mimic_metadata 
                ORDER BY table_name
            ''')
            
            tables = []
            for row in cursor.fetchall():
                tables.append({
                    'table_name': row[0],
                    'description': row[1],
                    'row_count': row[2],
                    'column_count': row[3]
                })
            
            conn.close()
            return tables
            
        except Exception as e:
            print(f"Error getting MIMIC tables: {e}")
            return []
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get all available datasets (uploaded + MIMIC-III)."""
        datasets = {
            'uploaded': [],
            'mimic': [],
            'has_uploaded': False,
            'has_mimic': False
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get uploaded datasets
            try:
                cursor.execute('''
                    SELECT filename, table_name, row_count, column_count, upload_date
                    FROM file_metadata 
                    WHERE data_source = 'uploaded'
                    ORDER BY upload_date DESC
                ''')
                
                for row in cursor.fetchall():
                    datasets['uploaded'].append({
                        'filename': row[0],
                        'table_name': row[1],
                        'row_count': row[2],
                        'column_count': row[3],
                        'upload_date': row[4]
                    })
            except Exception as e:
                print(f"Error getting uploaded datasets: {e}")
            
            datasets['has_uploaded'] = len(datasets['uploaded']) > 0
            
            # Get MIMIC-III datasets
            datasets['mimic'] = self.get_mimic_tables()
            datasets['has_mimic'] = len(datasets['mimic']) > 0
            
            conn.close()
            
        except Exception as e:
            print(f"Error getting available datasets: {e}")
        
        return datasets
    
    def clear_mimic_data(self) -> Dict[str, Any]:
        """Clear all MIMIC-III data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of MIMIC tables to drop
            mimic_tables = self.get_mimic_tables()
            
            # Drop each MIMIC table
            for table in mimic_tables:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS `{table['table_name']}`")
                except Exception as e:
                    print(f"Error dropping table {table['table_name']}: {e}")
            
            # Clear metadata
            cursor.execute("DELETE FROM mimic_metadata")
            cursor.execute("DELETE FROM file_metadata WHERE data_source = 'mimic'")
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': f'Cleared {len(mimic_tables)} MIMIC-III tables from database'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error clearing MIMIC data: {str(e)}'
            }
    
    def get_table_info(self, table_name: str) -> Tuple[List[Dict], List[Tuple], int]:
        """Get column information, sample data, and row count for a table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get column info
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = [{'name': row[1], 'type': row[2]} for row in cursor.fetchall()]
            
            # Get sample data
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
            sample_rows = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
            
            return columns, sample_rows, row_count
            
        except Exception as e:
            print(f"Error getting table info: {e}")
            return [], [], 0
        finally:
            conn.close()
    
    def execute_query(self, sql_query: str, table_name: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute SQL query safely with error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Basic security check
            sql_lower = sql_query.lower()
            dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', 'truncate']
            if any(keyword in sql_lower for keyword in dangerous_keywords):
                return None, "Query contains potentially dangerous operations and was blocked for security."
            
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names and results
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            conn.close()
            
            return {'columns': columns, 'rows': rows}, None
            
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            
            # Provide user-friendly error messages
            if "no such column" in error_msg:
                return None, "Column not found in the dataset. Please check the available column names and try again."
            elif "unrecognized token" in error_msg:
                return None, "SQL syntax error - possibly due to special characters in column names. Try rephrasing your question."
            elif "no such table" in error_msg:
                return None, "Dataset table not found. Please upload data first or select a valid table."
            else:
                return None, f"Database error: {error_msg}"
                
        except Exception as e:
            return None, f"Unexpected error executing query: {str(e)}"
    
    def save_query_history(self, table_name: str, question: str, sql_query: str, response: str):
        """Save query to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO query_history (table_name, question, sql_query, response)
                VALUES (?, ?, ?, ?)
            ''', (table_name, question, sql_query, response))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving query history: {e}")
    
    def get_basic_table_data(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get basic table data for dashboard display."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get basic info
            cursor = conn.cursor()
            cursor.execute(f'SELECT COUNT(*) FROM `{table_name}`')
            row_count = cursor.fetchone()[0]
            
            # Get column names
            cursor.execute(f'PRAGMA table_info(`{table_name}`)')
            columns = [row[1] for row in cursor.fetchall()]
            
            # Get sample data (first 5 rows)
            df_sample = pd.read_sql_query(f'SELECT * FROM `{table_name}` LIMIT 5', conn)
            
            # Get data source info
            cursor.execute('''
                SELECT data_source FROM file_metadata 
                WHERE table_name = ? 
                LIMIT 1
            ''', (table_name,))
            source_result = cursor.fetchone()
            data_source = source_result[0] if source_result else 'unknown'
            
            # Get description if it's MIMIC-III
            description = None
            if data_source == 'mimic':
                cursor.execute('''
                    SELECT description FROM mimic_metadata 
                    WHERE table_name = ?
                ''', (table_name,))
                desc_result = cursor.fetchone()
                description = desc_result[0] if desc_result else None
            
            conn.close()
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': columns,
                'sample_data': df_sample.to_html(classes='table table-striped', index=False),
                'data_source': data_source,
                'description': description
            }
            
        except Exception as e:
            print(f"Error loading basic table data: {e}")
            return None
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            result = cursor.fetchone() is not None
            conn.close()
            return result
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False
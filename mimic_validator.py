"""
MIMIC-III Data Validation and Setup Utility
Validates MIMIC-III CSV files and provides setup assistance
"""

import os
import sys
import pandas as pd
import glob
from typing import Dict, List, Tuple, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from application import DatabaseManager
except ImportError:
    print("Warning: Could not import DatabaseManager from application module")
    DatabaseManager = None

class MimicValidator:
    def __init__(self, mimic_data_path: str = "data/mimic_iii"):
        self.mimic_data_path = mimic_data_path
        
        # Expected MIMIC-III tables with their key characteristics
        self.expected_tables = {
            'admissions': {
                'required_columns': ['subject_id', 'hadm_id', 'admittime', 'dischtime'],
                'description': 'Hospital admissions data'
            },
            'patients': {
                'required_columns': ['subject_id', 'gender', 'dob'],
                'description': 'Patient demographics and information'
            },
            'icustays': {
                'required_columns': ['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime'],
                'description': 'ICU stay records'
            },
            'chartevents': {
                'required_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value'],
                'description': 'Chart events and vital signs'
            },
            'labevents': {
                'required_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value'],
                'description': 'Laboratory test results'
            },
            'prescriptions': {
                'required_columns': ['subject_id', 'hadm_id', 'drug', 'startdate'],
                'description': 'Medication prescriptions'
            },
            'diagnoses_icd': {
                'required_columns': ['subject_id', 'hadm_id', 'icd9_code'],
                'description': 'ICD diagnosis codes'
            },
            'procedures_icd': {
                'required_columns': ['subject_id', 'hadm_id', 'icd9_code'],
                'description': 'ICD procedure codes'
            },
            'noteevents': {
                'required_columns': ['subject_id', 'hadm_id', 'charttime', 'category', 'text'],
                'description': 'Clinical notes and reports'
            },
            'inputevents_cv': {
                'required_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'amount'],
                'description': 'Input events (CareVue)'
            },
            'inputevents_mv': {
                'required_columns': ['subject_id', 'hadm_id', 'itemid', 'starttime', 'amount'],
                'description': 'Input events (MetaVision)'
            },
            'outputevents': {
                'required_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value'],
                'description': 'Output events'
            },
            'd_items': {
                'required_columns': ['itemid', 'label', 'category'],
                'description': 'Items dictionary'
            },
            'd_labitems': {
                'required_columns': ['itemid', 'label', 'category'],
                'description': 'Lab items dictionary'
            },
            'd_icd_diagnoses': {
                'required_columns': ['icd9_code', 'short_title', 'long_title'],
                'description': 'ICD diagnosis dictionary'
            }
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Comprehensive validation of MIMIC-III setup."""
        print("🏥 MIMIC-III Data Validation Starting...")
        print("=" * 50)
        
        validation_results = {
            'directory_exists': False,
            'csv_files_found': [],
            'valid_tables': [],
            'invalid_tables': [],
            'missing_tables': [],
            'total_size_mb': 0,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if directory exists
        if not os.path.exists(self.mimic_data_path):
            validation_results['errors'].append(f"MIMIC-III data directory not found: {self.mimic_data_path}")
            print(f"❌ Directory not found: {self.mimic_data_path}")
            return validation_results
        
        validation_results['directory_exists'] = True
        print(f"✅ Directory found: {self.mimic_data_path}")
        
        # Find CSV files
        csv_files = glob.glob(os.path.join(self.mimic_data_path, "*.csv"))
        validation_results['csv_files_found'] = [os.path.basename(f) for f in csv_files]
        
        print(f"📁 Found {len(csv_files)} CSV files")
        
        if not csv_files:
            validation_results['errors'].append("No CSV files found in MIMIC-III directory")
            print("❌ No CSV files found")
            return validation_results
        
        # Validate each file
        total_size = 0
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
            total_size += file_size
            
            filename = os.path.basename(csv_file)
            table_name = filename.replace('.csv', '').lower()
            
            print(f"\n📊 Validating {filename} ({file_size:.1f} MB)...")
            
            validation_result = self._validate_table(csv_file, table_name)
            
            if validation_result['valid']:
                validation_results['valid_tables'].append({
                    'filename': filename,
                    'table_name': table_name,
                    'size_mb': file_size,
                    **validation_result
                })
                print(f"   ✅ Valid - {validation_result['row_count']} rows, {validation_result['column_count']} columns")
            else:
                validation_results['invalid_tables'].append({
                    'filename': filename,
                    'table_name': table_name,
                    'size_mb': file_size,
                    'errors': validation_result.get('errors', [])
                })
                print(f"   ❌ Invalid - {'; '.join(validation_result.get('errors', ['Unknown error']))}")
        
        validation_results['total_size_mb'] = round(total_size, 1)
        
        # Check for missing key tables
        found_table_names = {t['table_name'] for t in validation_results['valid_tables']}
        key_tables = {'patients', 'admissions', 'icustays'}
        missing_key_tables = key_tables - found_table_names
        
        if missing_key_tables:
            validation_results['warnings'].append(f"Missing key tables: {', '.join(missing_key_tables)}")
        
        # Generate recommendations
        self._generate_recommendations(validation_results)
        
        # Print summary
        self._print_summary(validation_results)
        
        return validation_results
    
    def _validate_table(self, csv_file: str, table_name: str) -> Dict[str, Any]:
        """Validate a single MIMIC-III table."""
        result = {
            'valid': False,
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Read first few rows to check structure
            df_sample = pd.read_csv(csv_file, nrows=100, low_memory=False)
            
            result['column_count'] = len(df_sample.columns)
            result['columns'] = df_sample.columns.tolist()
            
            # Get full row count efficiently
            with open(csv_file, 'r') as f:
                result['row_count'] = sum(1 for line in f) - 1  # Subtract header
            
            # Check if this is a known MIMIC-III table
            if table_name in self.expected_tables:
                expected = self.expected_tables[table_name]
                missing_columns = []
                
                # Check for required columns (case-insensitive)
                df_columns_lower = [col.lower().strip() for col in df_sample.columns]
                
                for req_col in expected['required_columns']:
                    if req_col.lower() not in df_columns_lower:
                        missing_columns.append(req_col)
                
                if missing_columns:
                    result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    result['valid'] = True
            else:
                # Unknown table - basic validation
                if result['column_count'] > 0 and result['row_count'] > 0:
                    result['valid'] = True
                    result['warnings'].append("Unknown MIMIC-III table - basic validation only")
                else:
                    result['errors'].append("Empty or invalid CSV file")
        
        except Exception as e:
            result['errors'].append(f"Failed to read CSV: {str(e)}")
        
        return result
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]):
        """Generate setup recommendations."""
        recommendations = validation_results['recommendations']
        
        if len(validation_results['valid_tables']) == 0:
            recommendations.append("❌ No valid tables found. Please check MIMIC-III data download and placement.")
        elif len(validation_results['valid_tables']) < 5:
            recommendations.append("⚠️  Few tables found. Consider downloading the complete MIMIC-III demo dataset.")
        else:
            recommendations.append("✅ Good table coverage for analysis.")
        
        if validation_results['total_size_mb'] > 1000:
            recommendations.append("💾 Large dataset detected. Loading may take several minutes.")
        
        # Check for key analysis tables
        table_names = {t['table_name'] for t in validation_results['valid_tables']}
        
        if 'patients' in table_names and 'admissions' in table_names:
            recommendations.append("🎯 Core demographic tables available - ready for basic analysis.")
        
        if 'chartevents' in table_names or 'labevents' in table_names:
            recommendations.append("📊 Clinical data tables available - ready for advanced analysis.")
        
        if len(validation_results['invalid_tables']) > 0:
            recommendations.append("🔧 Some files have issues. Check file formats and completeness.")
    
    def _print_summary(self, validation_results: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        
        print(f"✅ Valid tables: {len(validation_results['valid_tables'])}")
        print(f"❌ Invalid tables: {len(validation_results['invalid_tables'])}")
        print(f"💾 Total size: {validation_results['total_size_mb']} MB")
        
        if validation_results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in validation_results['warnings']:
                print(f"   {warning}")
        
        if validation_results['errors']:
            print(f"\n❌ ERRORS:")
            for error in validation_results['errors']:
                print(f"   {error}")
        
        if validation_results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_results['recommendations']:
                print(f"   {rec}")
        
        print("\n" + "=" * 50)
        
        if len(validation_results['valid_tables']) > 0:
            print("🚀 Ready to load MIMIC-III data in the application!")
        else:
            print("🔧 Please fix issues before loading data.")
    
    def get_table_suggestions(self) -> List[str]:
        """Get suggested tables for first-time users."""
        beginner_tables = [
            "patients - Start here for basic demographics",
            "admissions - Hospital admission details", 
            "icustays - ICU stay information",
            "diagnoses_icd - Patient diagnoses",
            "prescriptions - Medication data"
        ]
        return beginner_tables
    
    def get_sample_queries(self) -> Dict[str, List[str]]:
        """Generate sample queries for different tables."""
        queries = {
            'patients': [
                "How many patients are in the dataset?",
                "What's the age distribution of patients?",
                "Show me patients by gender"
            ],
            'admissions': [
                "What's the average length of stay?",
                "Which admission type is most common?",
                "Show me recent admissions"
            ],
            'icustays': [
                "How many ICU stays are there?",
                "What's the average ICU length of stay?",
                "Which ICU unit has the most stays?"
            ],
            'diagnoses_icd': [
                "What are the most common diagnoses?",
                "How many patients have diabetes?",
                "Show me cardiovascular diagnoses"
            ],
            'prescriptions': [
                "What are the most prescribed medications?",
                "How many prescriptions per patient on average?",
                "Show me pain medications"
            ]
        }
        return queries

def main():
    """Run MIMIC-III validation as standalone script."""
    print("🏥 MIMIC-III Data Profiling App - Validation Utility")
    print("=" * 60)
    
    validator = MimicValidator()
    results = validator.validate_setup()
    
    if len(results['valid_tables']) > 0:
        print(f"\n🎯 SUGGESTED STARTER TABLES:")
        for suggestion in validator.get_table_suggestions():
            print(f"   • {suggestion}")
        
        print(f"\n💬 SAMPLE QUESTIONS TO TRY:")
        sample_queries = validator.get_sample_queries()
        for table, queries in list(sample_queries.items())[:2]:  # Show first 2 tables
            if any(t['table_name'] == table for t in results['valid_tables']):
                print(f"   {table.upper()}:")
                for query in queries:
                    print(f"     - {query}")
        
        # Test database loading if DatabaseManager is available
        if DatabaseManager:
            print(f"\n🔗 TESTING DATABASE INTEGRATION:")
            try:
                db_manager = DatabaseManager()
                if db_manager.is_mimic_data_loaded():
                    print("   ✅ MIMIC-III data already loaded in database")
                else:
                    print("   ℹ️  MIMIC-III data not yet loaded - ready to load via app interface")
            except Exception as e:
                print(f"   ⚠️  Database test failed: {e}")

if __name__ == "__main__":
    main()
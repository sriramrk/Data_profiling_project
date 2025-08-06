#!/usr/bin/env python3
"""
MIMIC-III Setup Script for Data Profiling App
Simplified version that integrates with existing application structure
"""

import os
import sys
import shutil
import glob
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_directories():
    """Create necessary directories for MIMIC-III data."""
    directories = [
        "data",
        "data/mimic_iii", 
        "reports",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher is required")
    
    # Check required packages
    required_packages = ['flask', 'pandas', 'numpy', 'plotly']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check for .env file
    if not os.path.exists('.env'):
        issues.append("No .env file found - OpenAI features will be disabled")
    
    if issues:
        print("⚠️  Environment issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Environment looks good!")
        return True

def copy_mimic_files(source_path: str) -> bool:
    """Copy MIMIC-III files from source to app directory."""
    mimic_data_path = "data/mimic_iii"
    
    if not os.path.exists(source_path):
        print(f"❌ Source path not found: {source_path}")
        return False
    
    # Find CSV files in source
    source_files = glob.glob(os.path.join(source_path, "*.csv"))
    
    if not source_files:
        print(f"❌ No CSV files found in {source_path}")
        return False
    
    print(f"📁 Copying {len(source_files)} files from {source_path}...")
    
    for file_path in source_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(mimic_data_path, filename)
        shutil.copy2(file_path, dest_path)
        print(f"   📋 Copied {filename}")
    
    print(f"✅ Copied {len(source_files)} files successfully")
    return True

def validate_mimic_data() -> bool:
    """Validate MIMIC-III data setup."""
    print("\n🔍 Validating MIMIC-III data...")
    
    try:
        from mimic_validator import MimicValidator
        validator = MimicValidator()
        results = validator.validate_setup()
        return len(results['valid_tables']) > 0
    except ImportError:
        # Fallback validation if validator not available
        mimic_data_path = "data/mimic_iii"
        csv_files = glob.glob(os.path.join(mimic_data_path, "*.csv"))
        
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files")
            return True
        else:
            print("❌ No CSV files found")
            return False

def test_database_integration():
    """Test database integration."""
    print("\n💾 Testing database integration...")
    
    try:
        from application import DatabaseManager
        db_manager = DatabaseManager()
        
        # Test if we can load MIMIC data
        result = db_manager.load_mimic_data()
        
        if result['success']:
            print(f"✅ {result['message']}")
            
            if result.get('tables_loaded'):
                print(f"📊 Loaded tables:")
                for table in result['tables_loaded'][:5]:  # Show first 5
                    print(f"   • {table['table_name']}: {table['row_count']:,} rows")
                
                if len(result['tables_loaded']) > 5:
                    print(f"   ... and {len(result['tables_loaded']) - 5} more tables")
            
            return True
        else:
            print(f"❌ {result['message']}")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import DatabaseManager: {e}")
        return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    if not os.path.exists('.env'):
        env_content = """# Data Profiling App Configuration
# OpenAI API Key (required for AI features)
OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Database settings
DATABASE_PATH=data/app.db
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env template file")
        print("   Please edit .env and add your OpenAI API key for full functionality")
    else:
        print("✅ .env file already exists")

def print_next_steps(success: bool):
    """Print next steps for the user."""
    print("\n" + "="*60)
    
    if success:
        print("🎉 MIMIC-III SETUP COMPLETE!")
        print("="*60)
        
        print("\n📋 NEXT STEPS:")
        print("1. 🔑 Edit .env file and add your OpenAI API key")
        print("2. 🚀 Start the application: python app.py")
        print("3. 🌐 Open your browser to: http://127.0.0.1:5000")
        print("4. 🏥 Select 'MIMIC-III Demo Data' on the main page")
        print("5. 📊 Choose a table and start exploring!")
        
        print("\n💡 RECOMMENDED STARTER TABLES:")
        print("   • patients - Demographics and basic info")
        print("   • admissions - Hospital stays and outcomes")
        print("   • icustays - ICU stay details")
        print("   • diagnoses_icd - Patient diagnoses")
        print("   • prescriptions - Medication data")
        
        print("\n💬 SAMPLE QUESTIONS TO TRY:")
        print("   • 'How many patients are in the dataset?'")
        print("   • 'What is the average length of stay?'")
        print("   • 'Show me patients with diabetes'")
        print("   • 'What are the most common medications?'")
        
    else:
        print("❌ SETUP INCOMPLETE")
        print("="*60)
        
        print("\n🔧 TROUBLESHOOTING:")
        print("1. 📥 Download MIMIC-III demo data from:")
        print("   https://physionet.org/content/mimiciii-demo/1.4/")
        print("2. 📁 Extract all CSV files to a folder")
        print("3. 🔄 Run setup again with:")
        print("   python setup_mimic.py --source /path/to/mimic-iii-demo")
        
    print("\n🔍 VALIDATION & TROUBLESHOOTING:")
    print("   • Run: python mimic_validator.py")
    print("   • Check: python setup_mimic.py --validate")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup MIMIC-III data for the Data Profiling App')
    parser.add_argument('--source', help='Path to MIMIC-III CSV files')
    parser.add_argument('--validate', action='store_true', help='Only validate existing data')
    parser.add_argument('--skip-load', action='store_true', help='Skip loading data into database')
    
    args = parser.parse_args()
    
    print("🚀 MIMIC-III Data Profiling App - Setup")
    print("="*60)
    
    success = True
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before continuing.")
        print("   Install missing packages: pip install -r requirements.txt")
        success = False
    
    # Step 2: Create directories
    if success:
        print("\n📁 Creating directories...")
        create_directories()
    
    # Step 3: Create .env file
    if success:
        print("\n⚙️  Setting up configuration...")
        create_sample_env()
    
    # Step 4: Copy files if source provided
    if success and args.source and not args.validate:
        print(f"\n📥 Copying files from {args.source}...")
        success = copy_mimic_files(args.source)
    
    # Step 5: Validate data
    if success:
        success = validate_mimic_data()
    
    # Step 6: Load into database
    if success and not args.skip_load and not args.validate:
        success = test_database_integration()
    
    # Step 7: Show results
    print_next_steps(success)

if __name__ == "__main__":
    main()
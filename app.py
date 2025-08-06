"""
Enhanced Main Flask Application - Fixed MIMIC-III Support and Database Issues
"""

import os
import secrets
import re
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from dotenv import load_dotenv

# Import our custom modules
from application import DatabaseManager, ChatEngine, DataProfiler, ChartGenerator

# Load environment variables
load_dotenv()

# App configuration
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'data/app.db'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/mimic_iii', exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize managers
db_manager = DatabaseManager(DATABASE_PATH)

api_key = os.getenv('OPENAI_API_KEY')
profiler = DataProfiler(DATABASE_PATH, api_key)

chat_engine = ChatEngine(api_key) if api_key else None
chart_generator = ChartGenerator(DATABASE_PATH, api_key) if api_key else None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with dataset selection and current data display."""
    if request.method == 'POST':
        return handle_file_upload()
    
    # GET request - show dataset selection and current data
    try:
        datasets = db_manager.get_available_datasets()
    except Exception as e:
        print(f"Error getting datasets: {e}")
        datasets = {'uploaded': [], 'mimic': [], 'has_uploaded': False, 'has_mimic': False}
    
    current_table = session.get('current_table')
    current_mode = session.get('dataset_mode', 'none')
    data_info = None
    
    if current_table and db_manager.table_exists(current_table):
        try:
            data_info = db_manager.get_basic_table_data(current_table)
        except Exception as e:
            print(f"Error getting table data: {e}")
            # Clear invalid session data
            session.pop('current_table', None)
            session.pop('dataset_mode', None)
            current_table = None
            current_mode = 'none'
    
    return render_template('index.html', 
                         data_info=data_info,
                         datasets=datasets,
                         current_table=current_table,
                         current_mode=current_mode)

@app.route('/load-mimic', methods=['POST'])
def load_mimic_data():
    """Load MIMIC-III demo data with better error handling."""
    try:
        # Clear any existing session data
        session.pop('current_table', None)
        session.pop('dataset_mode', None)
        
        result = db_manager.load_mimic_data()
        
        if result['success']:
            session['dataset_mode'] = 'mimic'
            flash(result['message'], 'success')
            
            # Show available tables
            if result.get('tables_loaded'):
                table_count = len(result['tables_loaded'])
                
                # If already loaded, show existing tables
                if result.get('already_loaded'):
                    flash(f'MIMIC-III data ready! {table_count} tables available for analysis.', 'info')
                else:
                    # Newly loaded
                    table_list = ', '.join([t['table_name'] for t in result['tables_loaded'][:5]])
                    if table_count > 5:
                        table_list += f', and {table_count - 5} more'
                    flash(f'Successfully loaded: {table_list}', 'info')
                
                # Auto-select first table if available
                if result['tables_loaded']:
                    first_table = result['tables_loaded'][0]['table_name']
                    session['current_table'] = first_table
                    flash(f'Auto-selected table: {first_table}', 'success')
        else:
            flash(result['message'], 'error')
            
    except Exception as e:
        print(f"Error in load_mimic_data route: {e}")
        flash(f'Error loading MIMIC-III data: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/select-table/<table_name>')
def select_table(table_name):
    """Select a specific table for analysis."""
    try:
        if db_manager.table_exists(table_name):
            session['current_table'] = table_name
            
            # Determine dataset mode based on table source
            try:
                datasets = db_manager.get_available_datasets()
                if any(t['table_name'] == table_name for t in datasets['mimic']):
                    session['dataset_mode'] = 'mimic'
                elif any(t['table_name'] == table_name for t in datasets['uploaded']):
                    session['dataset_mode'] = 'uploaded'
                else:
                    session['dataset_mode'] = 'unknown'
            except Exception as e:
                print(f"Error determining dataset mode: {e}")
                session['dataset_mode'] = 'unknown'
            
            flash(f'Selected table: {table_name}', 'success')
        else:
            flash(f'Table {table_name} not found', 'error')
            
    except Exception as e:
        print(f"Error selecting table: {e}")
        flash(f'Error selecting table: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/mimic-tables')
def mimic_tables():
    """Show available MIMIC-III tables."""
    try:
        datasets = db_manager.get_available_datasets()
        return render_template('mimic_tables.html', 
                             mimic_tables=datasets['mimic'],
                             current_table=session.get('current_table'))
    except Exception as e:
        flash(f'Error loading MIMIC tables: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/clear-mimic', methods=['POST'])
def clear_mimic_data():
    """Clear MIMIC-III data from database."""
    try:
        result = db_manager.clear_mimic_data()
        
        if result['success']:
            # Clear session data if current table was MIMIC
            if session.get('dataset_mode') == 'mimic':
                session.pop('current_table', None)
                session.pop('dataset_mode', None)
            
            flash(result['message'], 'success')
        else:
            flash(result['message'], 'error')
            
    except Exception as e:
        flash(f'Error clearing MIMIC data: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reload-mimic', methods=['POST'])
def reload_mimic_data():
    """Reload MIMIC-III data (clear and reload)."""
    try:
        # First clear existing data
        clear_result = db_manager.clear_mimic_data()
        if not clear_result['success']:
            flash(f'Error clearing existing data: {clear_result["message"]}', 'error')
            return redirect(url_for('index'))
        
        # Then reload
        result = db_manager.load_mimic_data()
        
        if result['success']:
            session['dataset_mode'] = 'mimic'
            session.pop('current_table', None)  # Let user select table
            flash(f'Successfully reloaded MIMIC-III data: {result["message"]}', 'success')
            
            if result.get('tables_loaded'):
                table_count = len(result['tables_loaded'])
                flash(f'Reloaded {table_count} tables. Please select a table to begin analysis.', 'info')
        else:
            flash(result['message'], 'error')
            
    except Exception as e:
        flash(f'Error reloading MIMIC data: {str(e)}', 'error')
    
    return redirect(url_for('index'))

def handle_file_upload():
    """Handle CSV file upload process with better error handling."""
    # Validate file upload
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Only CSV files are allowed', 'error')
        return redirect(request.url)
    
    try:
        # Clear any existing session data
        session.pop('current_table', None)
        session.pop('dataset_mode', None)
        
        # Read CSV file with better handling of mixed types and encoding issues
        try:
            df = pd.read_csv(file, dtype=str, na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'], encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, dtype=str, na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'], encoding='latin-1')
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, dtype=str, na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'], encoding_errors='ignore')
        
        # Clean up any problematic data
        df = df.replace(to_replace=[r'\\t', r'\\n', r'\\r'], value=[' ', ' ', ' '], regex=True)
        
        # Convert numeric columns back to numeric where possible
        for col in df.columns:
            if col and not col.startswith('Unnamed'):
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() / len(df) > 0.7:
                    df[col] = numeric_series
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        
        if df.empty:
            flash('The CSV file is empty after cleaning', 'error')
            return redirect(request.url)
        
        # Limit the number of columns to prevent issues
        if len(df.columns) > 50:
            flash(f'Dataset has {len(df.columns)} columns. Using first 50 columns for performance.', 'warning')
            df = df.iloc[:, :50]
        
        # Generate table name from filename
        filename = secure_filename(file.filename)
        table_name = filename.rsplit('.', 1)[0].lower().replace(' ', '_').replace('-', '_')
        
        # Ensure table name is valid
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        if not table_name or table_name[0].isdigit():
            table_name = 'uploaded_' + table_name
        
        # Upload to database
        result = db_manager.upload_csv_data(df, filename, table_name)
        
        if result['success']:
            session['current_table'] = table_name
            session['dataset_mode'] = 'uploaded'
            flash(result['message'], 'success')
        else:
            flash(result['message'], 'error')
        
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"Error in file upload: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/chat')
def chat():
    """Chat interface with dataset info."""
    current_table = session.get('current_table')
    current_mode = session.get('dataset_mode', 'none')
    
    if not current_table:
        flash('Please select a dataset first', 'error')
        return redirect(url_for('index'))
    
    if not db_manager.table_exists(current_table):
        flash('Selected dataset not found', 'error')
        session.pop('current_table', None)
        session.pop('dataset_mode', None)
        return redirect(url_for('index'))
    
    # Get table info for display
    try:
        data_info = db_manager.get_basic_table_data(current_table)
    except Exception as e:
        print(f"Error getting table info: {e}")
        data_info = None
    
    return render_template('chat.html', 
                         table_name=current_table,
                         dataset_mode=current_mode,
                         data_info=data_info)

@app.route('/visualize')
def visualize():
    """Visualization interface with dataset info."""
    current_table = session.get('current_table')
    current_mode = session.get('dataset_mode', 'none')
    
    if not current_table:
        flash('Please select a dataset first', 'error')
        return redirect(url_for('index'))
    
    if not db_manager.table_exists(current_table):
        flash('Selected dataset not found', 'error')
        session.pop('current_table', None)
        session.pop('dataset_mode', None)
        return redirect(url_for('index'))
    
    return render_template('visualize.html', 
                         table_name=current_table,
                         dataset_mode=current_mode)

@app.route('/profile')
def profile_report():
    """Generate and display comprehensive profiling report."""
    current_table = session.get('current_table')
    
    if not current_table:
        flash('Please select a dataset first', 'error')
        return redirect(url_for('index'))
    
    if not db_manager.table_exists(current_table):
        flash('Selected dataset not found', 'error')
        session.pop('current_table', None)
        session.pop('dataset_mode', None)
        return redirect(url_for('index'))
    
    if profiler is None:
        flash('Profile generator not available', 'error')
        return redirect(url_for('index'))
    
    try:
        print(f"DEBUG: Starting profile generation for table: {current_table}")
        
        # Check if table has data
        import sqlite3
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM `{current_table}`")
        row_count = cursor.fetchone()[0]
        conn.close()
        
        if row_count == 0:
            flash('Dataset is empty - cannot generate profile', 'error')
            return redirect(url_for('index'))
        
        # Generate comprehensive profile
        profile_data = profiler.generate_comprehensive_profile(current_table)
        
        if profile_data is None:
            flash('Error: Profile generation returned no data', 'error')
            return redirect(url_for('index'))
        
        if not isinstance(profile_data, dict):
            flash(f'Error: Profile generation returned invalid data type: {type(profile_data)}', 'error')
            return redirect(url_for('index'))
        
        if 'error' in profile_data:
            flash(f'Error generating profile: {profile_data["error"]}', 'error')
            return redirect(url_for('index'))
        
        # Save HTML report
        report_path = profiler.save_profile_report(profile_data, current_table)
        
        print(f"DEBUG: Profile generation completed successfully")
        
        return render_template('report.html', 
                             profile_data=profile_data,
                             report_path=report_path,
                             table_name=current_table,
                             dataset_mode=session.get('dataset_mode', 'unknown'))
        
    except Exception as e:
        print(f"DEBUG: Critical error in profile_report: {str(e)}")
        flash(f'Error generating profile report: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for contextual chat queries."""
    if not chat_engine:
        return jsonify({
            'response': 'Chat functionality is not available. Please check your OpenAI API key configuration.',
            'sql_query': None,
            'error': 'OpenAI API key not configured'
        }), 500
    
    current_table = session.get('current_table')
    if not current_table:
        return jsonify({'error': 'No dataset selected'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    conversation_context = data.get('conversation_context', [])
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get table information
        columns, sample_rows, row_count = db_manager.get_table_info(current_table)
        
        if not columns:
            return jsonify({
                'response': 'Error: Could not access the dataset information. Please try re-uploading your data.',
                'sql_query': None,
                'error': 'Table info unavailable'
            })
        
        # Process question using contextual chat engine
        result = chat_engine.process_contextual_question(
            question, 
            current_table, 
            columns, 
            row_count, 
            db_manager, 
            conversation_context
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'response': f'I encountered an unexpected error while processing your question: {str(e)}\n\nPlease try asking a simpler question or rephrase your request.',
            'sql_query': None,
            'error': str(e),
            'context_used': False
        }), 500

@app.route('/api/dataset-info')
def api_dataset_info():
    """API endpoint to get dataset information with source."""
    current_table = session.get('current_table')
    if not current_table:
        return jsonify({'error': 'No dataset selected'}), 400
    
    if not db_manager.table_exists(current_table):
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Get table information
        columns, sample_rows, row_count = db_manager.get_table_info(current_table)
        
        # Analyze column types
        column_info = []
        for col in columns:
            col_category = 'text'
            if 'int' in col['type'].lower() or 'real' in col['type'].lower() or 'numeric' in col['type'].lower():
                col_category = 'numeric'
            elif 'date' in col['type'].lower() or 'time' in col['type'].lower():
                col_category = 'datetime'
            
            column_info.append({
                'name': col['name'],
                'type': col['type'],
                'category': col_category
            })
        
        return jsonify({
            'table_name': current_table,
            'row_count': row_count,
            'column_count': len(columns),
            'columns': column_info,
            'sample_data': [dict(zip([col['name'] for col in columns], row)) for row in sample_rows[:5]],
            'dataset_mode': session.get('dataset_mode', 'unknown')
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting dataset info: {str(e)}'}), 500

@app.route('/api/chart-suggestions')
def api_chart_suggestions():
    """API endpoint to get AI chart suggestions."""
    if chart_generator is None:
        return jsonify({
            'error': 'Chart generation not available - OpenAI API key required',
            'suggested_charts': []
        })
    
    current_table = session.get('current_table')
    if not current_table:
        return jsonify({'error': 'No dataset selected'}), 400
    
    try:
        suggestions = chart_generator.get_chart_suggestions(current_table)
        return jsonify(suggestions)
        
    except Exception as e:
        return jsonify({'error': f'Error getting chart suggestions: {str(e)}'}), 500

@app.route('/api/create-chart', methods=['POST'])
def api_create_chart():
    """API endpoint to create a chart."""
    if chart_generator is None:
        return jsonify({'error': 'Chart generation not available - OpenAI API key required'}), 500
    
    current_table = session.get('current_table')
    if not current_table:
        return jsonify({'error': 'No dataset selected'}), 400
    
    try:
        chart_config = request.get_json()
        result = chart_generator.create_chart(current_table, chart_config)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error creating chart: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return "<h1>404 - Page Not Found</h1><p>The page you're looking for doesn't exist.</p><a href='/'>Return to Dashboard</a>", 404

@app.errorhandler(500)
def internal_error(error):
    return "<h1>500 - Server Error</h1><p>Something went wrong on our end.</p><a href='/'>Return to Dashboard</a>", 500

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY not found in .env file")
        print("   Chat and advanced features will not work without it.")
    else:
        print("✅ OpenAI API key found - Full functionality enabled")
    
    # Check for MIMIC-III data
    if db_manager.is_mimic_data_loaded():
        print("✅ MIMIC-III demo data is available")
    else:
        print("ℹ️  MIMIC-III demo data not loaded")
        print(f"   Place MIMIC-III CSV files in: {os.path.abspath('data/mimic_iii')}")
    
    print("🚀 Starting Enhanced Data Profiling App with MIMIC-III Support...")
    print(f"📊 Database: {DATABASE_PATH}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🏥 MIMIC-III data folder: data/mimic_iii")
    print("🌐 Visit: http://127.0.0.1:5000")
    
    app.run(debug=True)
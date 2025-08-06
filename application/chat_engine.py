"""
Enhanced Chat Engine - Fixed to show numbered results with contextual conversation support
"""

import openai
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatEngine:
    def __init__(self, api_key: str):
        if api_key:
            openai.api_key = api_key
        
        # Context detection keywords
        self.context_keywords = [
            'these', 'them', 'those', 'above', 'previous', 'last', 'earlier',
            'from these', 'for these', 'in these', 'of these', 'with these',
            'from the above', 'from previous', 'from last', 'from earlier',
            'the ones', 'same records', 'same data', 'that result', 'that data'
        ]
        
        # Response templates for different query types
        self.response_templates = {
            'count': {
                'intro': 'I found the count you requested.',
                'method': 'I counted the records using SQL aggregation',
                'context_questions': ['What does this count represent?', 'Are there any filters that might affect this number?']
            },
            'average': {
                'intro': 'I calculated the average value for you.',
                'method': 'I computed the mean using SQL AVG function',
                'context_questions': ['Is this affected by outliers?', 'What does this average tell us about the data?']
            },
            'maximum': {
                'intro': 'I found the maximum value in the dataset.',
                'method': 'I used SQL MAX function to find the highest value',
                'context_questions': ['Is this value an outlier?', 'What record contains this maximum value?']
            },
            'minimum': {
                'intro': 'I found the minimum value in the dataset.',
                'method': 'I used SQL MIN function to find the lowest value',
                'context_questions': ['Is this value reasonable?', 'What record contains this minimum value?']
            },
            'group_analysis': {
                'intro': 'I analyzed the data by groups as requested.',
                'method': 'I used SQL GROUP BY to aggregate data by categories',
                'context_questions': ['What patterns do you notice?', 'Which groups are most/least represented?']
            },
            'correlation': {
                'intro': 'I analyzed the relationship between the variables.',
                'method': 'I calculated correlation using statistical functions',
                'context_questions': ['What might explain this relationship?', 'Should we investigate causation?']
            },
            'data_overview': {
                'intro': 'Here\'s an overview of your data.',
                'method': 'I queried the database structure and sample records',
                'context_questions': ['What specific aspect would you like to explore?', 'Are there any data quality concerns?']
            }
        }
    
    def escape_column_name(self, col_name: str) -> str:
        """Escape column names that might cause SQL issues."""
        if re.search(r'[^a-zA-Z0-9_]', col_name):
            return f'`{col_name}`'
        return col_name
    
    def detect_context_need(self, question: str) -> bool:
        """Detect if the question references previous context."""
        question_lower = question.lower()
        
        # Check for context keywords
        for keyword in self.context_keywords:
            if keyword in question_lower:
                return True
        
        # Check for patterns that suggest context reference
        context_patterns = [
            r'\b(for|from|in|of|with)\s+(these|them|those|above|previous|last)\b',
            r'\b(the\s+)?(same|that|those)\s+(records?|data|results?|ones?)\b',
            r'\bnow\s+(show|find|get|calculate|analyze)\b',
            r'\balso\s+(show|find|get|calculate|analyze)\b'
        ]
        
        for pattern in context_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _get_mimic_context(self, table_name: str, columns: List[Dict]) -> str:
        """Get MIMIC-III specific context for better query generation."""
        # MIMIC-III table descriptions and common patterns
        mimic_tables = {
            'patients': {
                'description': 'Patient demographics (subject_id is primary key)',
                'common_columns': ['subject_id', 'gender', 'dob', 'dod'],
                'tips': 'Use subject_id to link with other tables. Calculate age using dob.'
            },
            'admissions': {
                'description': 'Hospital admissions (hadm_id is admission ID)',
                'common_columns': ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type'],
                'tips': 'Calculate length of stay using dischtime - admittime. Link to patients via subject_id.'
            },
            'icustays': {
                'description': 'ICU stays (icustay_id is ICU stay ID)',
                'common_columns': ['subject_id', 'hadm_id', 'icustay_id', 'intime', 'outtime', 'los'],
                'tips': 'los is length of stay in days. Link to admissions via hadm_id.'
            },
            'chartevents': {
                'description': 'Vital signs and chart data (itemid refers to measurement type)',
                'common_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum'],
                'tips': 'Use valuenum for numeric analysis. itemid identifies what was measured.'
            },
            'labevents': {
                'description': 'Laboratory test results (itemid refers to lab test type)',
                'common_columns': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum'],
                'tips': 'Use valuenum for numeric lab values. Link with d_labitems for test names.'
            },
            'prescriptions': {
                'description': 'Medication prescriptions',
                'common_columns': ['subject_id', 'hadm_id', 'drug', 'dose_val_rx', 'startdate', 'enddate'],
                'tips': 'drug contains medication names. Use LIKE for partial matches.'
            },
            'diagnoses_icd': {
                'description': 'ICD-9 diagnosis codes',
                'common_columns': ['subject_id', 'hadm_id', 'icd9_code', 'short_title', 'long_title'],
                'tips': 'Link with d_icd_diagnoses for diagnosis descriptions. Use LIKE for code patterns.'
            },
            'procedures_icd': {
                'description': 'ICD-9 procedure codes',
                'common_columns': ['subject_id', 'hadm_id', 'icd9_code', 'short_title', 'long_title'],
                'tips': 'Link with d_icd_procedures for procedure descriptions.'
            }
        }
        
        # Check if this looks like a MIMIC-III table
        table_key = table_name.lower()
        if table_key in mimic_tables:
            info = mimic_tables[table_key]
            return f"\n\nMIMIC-III TABLE CONTEXT:\n- {info['description']}\n- Key insight: {info['tips']}\n"
        
        # Check for common MIMIC-III patterns
        column_names = [col['name'].lower() for col in columns]
        
        if 'subject_id' in column_names:
            return f"\n\nMIMIC-III CONTEXT:\n- This appears to be MIMIC-III clinical data\n- subject_id is the patient identifier\n- Use subject_id to join with other patient tables\n"
        elif 'hadm_id' in column_names:
            return f"\n\nMIMIC-III CONTEXT:\n- This appears to be MIMIC-III clinical data\n- hadm_id is the hospital admission identifier\n- Use hadm_id to join with admission-related tables\n"
        elif 'itemid' in column_names:
            return f"\n\nMIMIC-III CONTEXT:\n- This appears to be MIMIC-III clinical data\n- itemid refers to specific measurements or items\n- Consider linking with dictionary tables (d_items, d_labitems) for descriptions\n"
        
        return ""
    
    def classify_query_type(self, question: str, sql_query: str) -> str:
        """Classify the type of query to use appropriate response template."""
        question_lower = question.lower()
        sql_lower = sql_query.lower() if sql_query else ""
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            return 'count'
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            return 'average'
        elif any(word in question_lower for word in ['maximum', 'max', 'highest', 'largest']):
            return 'maximum'
        elif any(word in question_lower for word in ['minimum', 'min', 'lowest', 'smallest']):
            return 'minimum'
        elif 'group by' in sql_lower or any(word in question_lower for word in ['by', 'per', 'each']):
            return 'group_analysis'
        elif any(word in question_lower for word in ['correlation', 'relationship', 'related']):
            return 'correlation'
        elif any(word in question_lower for word in ['overview', 'summary', 'describe', 'show me']):
            return 'data_overview'
        else:
            return 'general'
    
    def generate_contextual_sql_query(self, question: str, table_name: str, columns: List[Dict], 
                                    row_count: int, context: List[Dict] = None) -> Optional[str]:
        """Generate SQL query with conversation context awareness."""
        
        # Create detailed column description
        column_descriptions = []
        for col in columns:
            escaped_name = self.escape_column_name(col['name'])
            col_info = f"{escaped_name} ({col['type']})"
            column_descriptions.append(col_info)
        
        column_list = ", ".join(column_descriptions)
        mimic_context = self._get_mimic_context(table_name, columns)
        # Check if context is needed
        needs_context = self.detect_context_need(question)
        
        # Build context information
        context_info = ""
        if needs_context and context:
            context_info = "\n\nCONVERSATION CONTEXT:\n"
            context_info += "Previous questions and queries in this conversation:\n"
            
            for i, ctx in enumerate(context[-3:], 1):  # Last 3 interactions
                context_info += f"{i}. Question: \"{ctx.get('question', 'N/A')}\"\n"
                context_info += f"   SQL Used: {ctx.get('sql_query', 'N/A')}\n"
                context_info += f"   Results: {ctx.get('result_count', 0)} rows returned\n"
                
                # Add sample results if available
                if ctx.get('sample_results'):
                    context_info += f"   Sample data: {ctx['sample_results'][:200]}...\n"
                context_info += "\n"
            
            context_info += "IMPORTANT: When the user refers to 'these', 'them', 'those', 'above', 'previous', etc., "
            context_info += "they are likely referring to the results from the most recent query above. "
            context_info += "You may need to use the previous query as a subquery or reference its conditions.\n"
        
        # Enhanced prompt with context awareness
        prompt = f"""
You are an expert SQL analyst working with a SQLite database. Generate an optimized SQL query to answer the user's question.

DATABASE CONTEXT:
- Table name: {table_name}
- Available columns: {column_list}
- Total rows: {row_count:,}
- Database type: SQLite{context_info}

IMPORTANT RULES:
1. Use backticks (`) around column names with special characters or spaces
2. For queries that might return many rows, use LIMIT 10 unless user asks for all
3. When comparing values, be case-insensitive using LOWER() when appropriate
4. For percentage calculations, multiply by 100.0 to avoid integer division
5. Use meaningful aliases for calculated columns
6. Order results logically (usually DESC for rankings, ASC for time series)
7. Handle NULL values appropriately
8. If the question references previous results (using words like 'these', 'them', 'those', 'above'), 
   incorporate the previous query conditions or use it as a subquery

CONTEXT-AWARE EXAMPLES:
- If previous query was "SELECT * FROM table WHERE bmi < 20 ORDER BY bmi LIMIT 10"
  and current question is "show me the average charges for these"
  then generate: "SELECT AVG(charges) as average_charges FROM {table_name} WHERE bmi < 20"

- If previous query selected specific records and current question asks about "those records",
  use the same WHERE conditions or LIMIT from the previous query

USER QUESTION: "{question}"

Generate only the SQL query (no explanations, no markdown formatting):
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the query
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```', '', sql_query)
            sql_query = sql_query.strip()
            
            # Validate query safety
            if self._is_query_safe(sql_query):
                return sql_query
            else:
                return None
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None
    
    def generate_sql_query(self, question: str, table_name: str, columns: List[Dict], row_count: int) -> Optional[str]:
        """Enhanced SQL generation with better error handling and optimization."""
        # This method now calls the contextual version for backward compatibility
        return self.generate_contextual_sql_query(question, table_name, columns, row_count, None)
    
    def _is_query_safe(self, sql_query: str) -> bool:
        """Validate that the SQL query is safe to execute."""
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', 'truncate', 'grant', 'revoke']
        sql_lower = sql_query.lower()
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        return True
    
    def generate_response(self, question: str, query_result: Dict, sql_query: str, 
                         has_context: bool = False) -> str:
        """Generate enhanced structured response with context awareness."""
        
        if not query_result or not query_result.get('rows'):
            return self._handle_no_results(question, sql_query)
        
        # Classify query type for appropriate response structure
        query_type = self.classify_query_type(question, sql_query)
        
        # Analyze the results
        analysis = self._analyze_query_results(query_result, query_type)
        
        # Generate structured response
        response_structure = {
            'answer': self._generate_direct_answer(query_result, question, query_type, has_context),
            'explanation': self._generate_explanation(sql_query, query_type, analysis, has_context),
            'insights': self._generate_insights(query_result, analysis, query_type),
            'context': self._generate_context(analysis, has_context),
            'follow_up_suggestions': self._generate_follow_up_questions(query_type, analysis),
            'data_sample': self._format_data_sample(query_result),
            'methodology': self._explain_methodology(sql_query, query_type)
        }
        
        # Format as readable text
        formatted_response = self._format_response_text(response_structure)
        
        return formatted_response
    
    def _handle_no_results(self, question: str, sql_query: str) -> str:
        """Handle cases where no results are returned."""
        return f"""
🔍 **No Results Found**

I executed your query but didn't find any matching data. This could mean:

• The specific criteria you're looking for doesn't exist in the dataset
• The column names or values might be slightly different than expected  
• The data might be filtered out by conditions in your question
• If you're referencing previous results, the context might not match

**What I tried:**
I used this SQL query: `{sql_query}`

**Suggestions:**
• Try rephrasing your question with different terms
• Ask me to show you the available column names and sample data
• Try a broader search without specific filters
• If referencing previous results, be more specific about what you want

Would you like me to help you explore what data is available instead?
"""
    
    def _analyze_query_results(self, query_result: Dict, query_type: str) -> Dict[str, Any]:
        """Analyze query results to provide insights."""
        columns = query_result['columns']
        rows = query_result['rows']
        
        analysis = {
            'result_count': len(rows),
            'column_count': len(columns),
            'columns': columns,
            'has_numeric_data': False,
            'has_categorical_data': False,
            'value_range': None,
            'top_values': [],
            'data_types': []
        }
        
        if len(rows) > 0 and len(columns) > 0:
            # Analyze first column for patterns
            first_col_values = [row[0] for row in rows if row[0] is not None]
            
            if first_col_values:
                # Check if numeric
                try:
                    numeric_values = [float(val) for val in first_col_values if isinstance(val, (int, float))]
                    if numeric_values:
                        analysis['has_numeric_data'] = True
                        analysis['value_range'] = {
                            'min': min(numeric_values),
                            'max': max(numeric_values),
                            'avg': sum(numeric_values) / len(numeric_values)
                        }
                except:
                    pass
                
                # Get top values for categorical analysis
                if len(set(first_col_values)) <= 10:  # Likely categorical
                    analysis['has_categorical_data'] = True
                    from collections import Counter
                    value_counts = Counter(first_col_values)
                    analysis['top_values'] = list(value_counts.most_common(5))
        
        return analysis
    
    def _generate_direct_answer(self, query_result: Dict, question: str, query_type: str, 
                              has_context: bool = False) -> str:
        """Generate the direct answer to the question with context awareness."""
        rows = query_result['rows']
        columns = query_result['columns']
        
        # Add context indicator if applicable
        context_prefix = "**Based on your previous query:** " if has_context else ""
        
        if len(rows) == 1 and len(columns) == 1:
            # Single value result
            value = rows[0][0]
            if value is None:
                return context_prefix + "No data found matching your criteria."
            
            # Format based on query type
            if query_type == 'count':
                return context_prefix + f"**{value:,}** records match your criteria."
            elif query_type in ['average', 'mean']:
                return context_prefix + f"The average is **{value:.2f}**."
            elif query_type == 'maximum':
                return context_prefix + f"The maximum value is **{value}**."
            elif query_type == 'minimum':
                return context_prefix + f"The minimum value is **{value}**."
            else:
                return context_prefix + f"The result is **{value}**."
        
        elif len(rows) == 1:
            # Single row, multiple columns
            result_dict = dict(zip(columns, rows[0]))
            formatted_items = [f"**{k}**: {v}" for k, v in result_dict.items() if v is not None]
            return context_prefix + "Here's what I found: " + ", ".join(formatted_items)
        
        else:
            # Multiple rows - format with numbered lines
            if len(rows) <= 10:  # Show detailed results for reasonable number of rows
                result_lines = []
                if context_prefix:
                    result_lines.append(context_prefix.rstrip())
                
                for i, row in enumerate(rows, 1):
                    row_dict = dict(zip(columns, row))
                    row_items = [f"{k}: {v}" for k, v in row_dict.items() if v is not None]
                    result_lines.append(f"{i}. {', '.join(row_items)}")
                
                return "\n".join(result_lines)
            else:
                # For large result sets, show count with note about sample
                if query_type == 'group_analysis':
                    return context_prefix + f"I found **{len(rows)}** different groups in your analysis."
                else:
                    return context_prefix + f"Your query returned **{len(rows)}** results. Showing sample in details below."
    
    def _generate_explanation(self, sql_query: str, query_type: str, analysis: Dict, 
                            has_context: bool = False) -> str:
        """Generate explanation of how the result was found with context awareness."""
        base_explanation = self.response_templates.get(query_type, {}).get('method', 'I analyzed your data using SQL')
        
        if has_context:
            base_explanation = "Building on your previous question, " + base_explanation.lower()
        
        # Add query-specific details
        if 'GROUP BY' in sql_query.upper():
            explanation = f"{base_explanation}. I grouped the data by categories and calculated aggregates for each group."
        elif 'ORDER BY' in sql_query.upper():
            explanation = f"{base_explanation}. I sorted the results to show the most relevant information first."
        elif 'WHERE' in sql_query.upper():
            explanation = f"{base_explanation}. I applied filters to find only the data matching your criteria."
        else:
            explanation = base_explanation
        
        # Add data scope information
        if analysis['result_count'] > 1:
            explanation += f" The analysis covers {analysis['result_count']} data points."
        
        return explanation
    
    def _generate_insights(self, query_result: Dict, analysis: Dict, query_type: str) -> List[str]:
        """Generate insights based on the query results."""
        insights = []
        
        # Numeric insights
        if analysis['has_numeric_data'] and analysis['value_range']:
            value_range = analysis['value_range']
            spread = value_range['max'] - value_range['min']
            insights.append(f"Values range from {value_range['min']:.2f} to {value_range['max']:.2f} (spread: {spread:.2f})")
            
            if value_range['avg']:
                if value_range['avg'] > (value_range['min'] + value_range['max']) / 2:
                    insights.append("Average is above the midpoint, suggesting some higher values pull it up")
                else:
                    insights.append("Average is below the midpoint, suggesting some lower values pull it down")
        
        # Categorical insights
        if analysis['has_categorical_data'] and analysis['top_values']:
            top_value, top_count = analysis['top_values'][0]
            total_results = analysis['result_count']
            percentage = (top_count / total_results) * 100 if total_results > 0 else 0
            insights.append(f"'{top_value}' is the most common value, appearing {percentage:.1f}% of the time")
            
            if len(analysis['top_values']) > 1:
                insights.append(f"Top {len(analysis['top_values'])} values represent the most frequent categories")
        
        # Data distribution insights
        if analysis['result_count'] == 1:
            insights.append("This is a single, specific result")
        elif analysis['result_count'] <= 5:
            insights.append("Small result set - these are likely the most relevant matches")
        elif analysis['result_count'] <= 20:
            insights.append("Moderate result set - good sample for analysis")
        else:
            insights.append("Large result set - showing top results (use LIMIT in query)")
        
        return insights
    
    def _generate_context(self, analysis: Dict, has_context: bool = False) -> str:
        """Generate context about the data and analysis."""
        context_parts = []
        
        if has_context:
            context_parts.append("This analysis builds on your previous question")
        
        if analysis['result_count'] > 0:
            context_parts.append(f"Analysis based on {analysis['result_count']} record(s)")
        
        if analysis['has_numeric_data']:
            context_parts.append("Contains numeric data suitable for mathematical analysis")
        
        if analysis['has_categorical_data']:
            context_parts.append("Contains categorical data good for grouping and comparison")
        
        return " • ".join(context_parts)
    
    def _generate_follow_up_questions(self, query_type: str, analysis: Dict) -> List[str]:
        """Generate suggested follow-up questions."""
        suggestions = []
        
        # Type-specific suggestions
        template_suggestions = self.response_templates.get(query_type, {}).get('context_questions', [])
        suggestions.extend(template_suggestions)
        
        # Data-driven suggestions
        if analysis['has_numeric_data']:
            suggestions.append("Would you like to see the distribution or outliers in this data?")
        
        if analysis['has_categorical_data']:
            suggestions.append("Would you like to see how these categories compare to each other?")
        
        if analysis['result_count'] > 10:
            suggestions.append("Would you like me to focus on the top results or apply additional filters?")
        
        # Context-aware suggestions
        suggestions.append("Want to analyze these results further?")
        suggestions.append("Should I compare this with other data?")
        
        # Limit to top 3 suggestions
        return suggestions[:3]
    
    def _format_data_sample(self, query_result: Dict) -> Dict[str, Any]:
        """Format a sample of the data for display."""
        columns = query_result['columns']
        rows = query_result['rows']
        
        if not rows:
            return {'message': 'No data to display'}
        
        # Show up to 5 rows
        sample_rows = rows[:5]
        formatted_data = []
        
        for row in sample_rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i] if i < len(row) else None
                # Format values for better display
                if isinstance(value, float):
                    row_dict[col] = round(value, 3)
                else:
                    row_dict[col] = value
            formatted_data.append(row_dict)
        
        return {
            'columns': columns,
            'data': formatted_data,
            'total_rows': len(rows),
            'showing': len(sample_rows)
        }
    
    def _explain_methodology(self, sql_query: str, query_type: str) -> str:
        """Explain the methodology used to answer the question."""
        methodology = f"**SQL Query Used:** `{sql_query}`\n\n"
        
        # Parse query components
        sql_upper = sql_query.upper()
        
        if 'SELECT' in sql_upper:
            methodology += "**Method:** Retrieved specific columns and calculated results using SQL\n"
        
        if 'WHERE' in sql_upper:
            methodology += "**Filtering:** Applied conditions to focus on relevant data\n"
        
        if 'GROUP BY' in sql_upper:
            methodology += "**Grouping:** Organized data by categories for comparison\n"
        
        if 'ORDER BY' in sql_upper:
            methodology += "**Sorting:** Arranged results in logical order\n"
        
        if 'LIMIT' in sql_upper:
            methodology += "**Limiting:** Focused on top results for clarity\n"
        
        return methodology
    
    def _format_response_text(self, structure: Dict[str, Any]) -> str:
        """Format the structured response into readable text."""
        response_parts = []
        
        # Main answer
        response_parts.append(f"📊 **Answer:** {structure['answer']}")
        
        # Explanation
        if structure.get('explanation'):
            response_parts.append(f"\n🔍 **How I found this:** {structure['explanation']}")
        
        # Key insights
        if structure.get('insights'):
            insights_text = '\n'.join([f"• {insight}" for insight in structure['insights']])
            response_parts.append(f"\n💡 **Key Insights:**\n{insights_text}")
        
        # Context
        if structure.get('context'):
            response_parts.append(f"\n📋 **Context:** {structure['context']}")
        
        # Data sample (if multiple results and not already shown in answer)
        if structure.get('data_sample') and structure['data_sample'].get('data'):
            sample = structure['data_sample']
            if sample['total_rows'] > 10:  # Only show sample if we have large result set
                response_parts.append(f"\n📈 **Sample Results** (showing {sample['showing']} of {sample['total_rows']}):")
                for i, row in enumerate(sample['data'][:3], 1):
                    row_text = ', '.join([f"{k}: {v}" for k, v in row.items() if v is not None])
                    response_parts.append(f"   {i}. {row_text}")
        
        # Follow-up suggestions
        if structure.get('follow_up_suggestions'):
            suggestions_text = '\n'.join([f"• {suggestion}" for suggestion in structure['follow_up_suggestions']])
            response_parts.append(f"\n🤔 **What's next?**\n{suggestions_text}")
        
        # Methodology (expandable section)
        if structure.get('methodology'):
            response_parts.append(f"\n<details><summary>🔧 <strong>Technical Details</strong></summary>\n{structure['methodology']}</details>")
        
        return '\n'.join(response_parts)
    
    def process_question(self, question: str, table_name: str, columns: List[Dict], row_count: int, db_manager) -> Dict[str, Any]:
        """Enhanced question processing with structured responses."""
        
        # Generate SQL query
        sql_query = self.generate_sql_query(question, table_name, columns, row_count)
        if not sql_query:
            return {
                'response': '❌ I couldn\'t generate a proper SQL query for your question. Please try rephrasing it or ask a simpler question about your data.\n\n💡 **Tip:** Try asking specific questions like "How many rows are there?" or "What are the column names?"',
                'sql_query': None,
                'error': 'Query generation failed'
            }
        
        # Execute query
        query_result, error = db_manager.execute_query(sql_query, table_name)
        if error:
            return {
                'response': f'⚠️ **Database Error:** {error}\n\n🔧 **SQL Query:** `{sql_query}`\n\n💡 **Tip:** Make sure you\'re using the exact column names from your dataset. Ask me to "show column names" if you need to see them.',
                'sql_query': sql_query,
                'error': error
            }
        
        # Generate enhanced response
        enhanced_response = self.generate_response(question, query_result, sql_query)
        
        # Save to history
        db_manager.save_query_history(table_name, question, sql_query, enhanced_response)
        
        return {
            'response': enhanced_response,
            'sql_query': sql_query,
            'data': query_result
        }
    
    def process_contextual_question(self, question: str, table_name: str, columns: List[Dict], 
                                  row_count: int, db_manager, conversation_context: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced question processing with conversation context support."""
        
        # Detect if context is needed
        needs_context = self.detect_context_need(question)
        
        # Generate contextual SQL query
        sql_query = self.generate_contextual_sql_query(
            question, table_name, columns, row_count, 
            conversation_context if needs_context else None
        )
        
        if not sql_query:
            return {
                'response': '❌ I couldn\'t generate a proper SQL query for your question. Please try rephrasing it or ask a simpler question about your data.\n\n💡 **Tip:** Try asking specific questions like "How many rows are there?" or "What are the column names?"',
                'sql_query': None,
                'error': 'Query generation failed',
                'context_used': False
            }
        
        # Execute query
        query_result, error = db_manager.execute_query(sql_query, table_name)
        if error:
            return {
                'response': f'⚠️ **Database Error:** {error}\n\n🔧 **SQL Query:** `{sql_query}`\n\n💡 **Tip:** Make sure you\'re using the exact column names from your dataset. Ask me to "show column names" if you need to see them.',
                'sql_query': sql_query,
                'error': error,
                'context_used': needs_context
            }
        
        # Generate enhanced response with context awareness
        enhanced_response = self.generate_response(question, query_result, sql_query, needs_context)
        
        # Save to history
        db_manager.save_query_history(table_name, question, sql_query, enhanced_response)
        
        # Prepare context for next interaction
        current_context = {
            'question': question,
            'sql_query': sql_query,
            'result_count': len(query_result.get('rows', [])),
            'columns': query_result.get('columns', []),
            'sample_results': str(query_result.get('rows', [])[:3]) if query_result.get('rows') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'response': enhanced_response,
            'sql_query': sql_query,
            'data': query_result,
            'context_used': needs_context,
            'current_context': current_context
        }
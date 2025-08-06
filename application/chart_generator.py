"""
Enhanced Chart Generator - Fixed Layout and Sizing Issues
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import numpy as np
from typing import Dict, List, Any, Optional

class ChartGenerator:
    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        
        # Chart type mapping for user-friendly names
        self.chart_types = {
            'bar': 'Bar Chart',
            'line': 'Line Chart', 
            'scatter': 'Scatter Plot',
            'histogram': 'Histogram',
            'box': 'Box Plot',
            'pie': 'Pie Chart',
            'heatmap': 'Heatmap',
            'violin': 'Violin Plot',
            'area': 'Area Chart'
        }
        
        # FIXED: Default layout settings for better chart display
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'size': 12, 'family': 'Arial, sans-serif'},
            'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
            'autosize': True,
            'showlegend': True,
            'legend': {'orientation': 'v', 'x': 1.02, 'xanchor': 'left'},
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }
    
    def get_chart_suggestions(self, table_name: str) -> Dict[str, Any]:
        """Get basic chart suggestions based on data analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000", conn)  # Limit for performance
            conn.close()
            
            if df.empty:
                return {'error': 'Dataset is empty', 'suggested_charts': []}
            
            # Analyze data types
            suggestions = self._get_default_suggestions(df)
            
            return {
                'suggested_charts': suggestions,
                'available_chart_types': self.chart_types,
                'data_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(df.select_dtypes(include=['object']).columns)
                }
            }
            
        except Exception as e:
            return {'error': f'Error analyzing data for charts: {str(e)}', 'suggested_charts': []}
    
    def _get_default_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Provide default chart suggestions based on data types."""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in df.columns if df[col].nunique() <= 20 and col not in numeric_cols]
        
        # Bar chart for categorical data with numeric data
        if categorical_cols and numeric_cols:
            suggestions.append({
                'chart_type': 'bar',
                'x_axis': categorical_cols[0],
                'y_axis': numeric_cols[0],
                'title': f"{numeric_cols[0]} by {categorical_cols[0]}",
                'explanation': 'Bar chart shows distribution of numeric values across categories',
                'insights': 'Compare values between different groups'
            })
        
        # Histogram for numeric distribution
        if numeric_cols:
            suggestions.append({
                'chart_type': 'histogram',
                'x_axis': numeric_cols[0],
                'y_axis': None,
                'title': f"Distribution of {numeric_cols[0]}",
                'explanation': 'Histogram shows the frequency distribution of numeric values',
                'insights': 'Understand data distribution, outliers, and central tendencies'
            })
        
        # Scatter plot for correlation
        if len(numeric_cols) >= 2:
            suggestions.append({
                'chart_type': 'scatter',
                'x_axis': numeric_cols[0],
                'y_axis': numeric_cols[1],
                'title': f"{numeric_cols[0]} vs {numeric_cols[1]}",
                'explanation': 'Scatter plot reveals relationships between two numeric variables',
                'insights': 'Identify correlations, patterns, and outliers'
            })
        
        # Pie chart for categorical distribution
        if categorical_cols:
            suggestions.append({
                'chart_type': 'pie',
                'x_axis': categorical_cols[0],
                'y_axis': None,
                'title': f"Distribution of {categorical_cols[0]}",
                'explanation': 'Pie chart shows proportional distribution of categories',
                'insights': 'See relative sizes of different categories'
            })
        
        return suggestions
    
    def create_chart(self, table_name: str, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chart based on user specifications."""
        try:
            # Load data
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            if df.empty:
                return {'error': 'Dataset is empty'}
            
            # Validate chart configuration
            validation_error = self._validate_chart_config(df, chart_config)
            if validation_error:
                return {'error': validation_error}
            
            # Generate chart based on type
            chart_type = chart_config.get('chart_type', '').lower()
            
            if chart_type == 'bar':
                fig = self._create_bar_chart(df, chart_config)
            elif chart_type == 'line':
                fig = self._create_line_chart(df, chart_config)
            elif chart_type == 'scatter':
                fig = self._create_scatter_plot(df, chart_config)
            elif chart_type == 'histogram':
                fig = self._create_histogram(df, chart_config)
            elif chart_type == 'box':
                fig = self._create_box_plot(df, chart_config)
            elif chart_type == 'pie':
                fig = self._create_pie_chart(df, chart_config)
            elif chart_type == 'heatmap':
                fig = self._create_heatmap(df, chart_config)
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}
            
            # FIXED: Apply enhanced layout settings
            fig.update_layout(self.default_layout)
            
            # Convert to JSON for frontend
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Generate basic insights
            insights = self._generate_basic_insights(df, chart_config)
            
            return {
                'success': True,
                'chart_json': chart_json,
                'chart_config': chart_config,
                'insights': insights,
                'data_summary': {
                    'rows_plotted': len(df),
                    'chart_type': self.chart_types.get(chart_type, chart_type)
                }
            }
            
        except Exception as e:
            return {'error': f'Error creating chart: {str(e)}'}
    
    def _validate_chart_config(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[str]:
        """Validate chart configuration against dataset."""
        chart_type = config.get('chart_type', '').lower()
        x_axis = config.get('x_axis')
        y_axis = config.get('y_axis')
        
        # Check if chart type is supported
        if chart_type not in self.chart_types:
            return f'Unsupported chart type: {chart_type}'
        
        # Check if required columns exist
        if x_axis and x_axis not in df.columns:
            return f'Column "{x_axis}" not found in dataset'
        
        if y_axis and y_axis not in df.columns:
            return f'Column "{y_axis}" not found in dataset'
        
        # Type-specific validations
        if chart_type in ['scatter', 'line'] and not y_axis:
            return f'{chart_type.title()} chart requires both X and Y axis columns'
        
        if chart_type == 'histogram' and not x_axis:
            return 'Histogram requires an X axis column'
        
        return None
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart with enhanced layout."""
        x_col = config['x_axis']
        y_col = config.get('y_axis')
        
        if y_col and y_col in df.columns:
            # Grouped bar chart
            fig = px.bar(df, x=x_col, y=y_col, title=config.get('title', f'{y_col} by {x_col}'))
        else:
            # Count of categories
            counts = df[x_col].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, 
                        title=config.get('title', f'Count of {x_col}'),
                        labels={'x': x_col, 'y': 'Count'})
        
        # FIXED: Enhance layout for better readability
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col if y_col else 'Count',
            height=500,
            showlegend=False if not y_col else True
        )
        
        # Rotate x-axis labels if many categories
        if len(df[x_col].unique()) > 10:
            fig.update_layout(xaxis={'tickangle': 45})
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create line chart with enhanced layout."""
        x_col = config['x_axis']
        y_col = config['y_axis']
        
        fig = px.line(df, x=x_col, y=y_col, title=config.get('title', f'{y_col} over {x_col}'))
        
        # FIXED: Enhanced layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot with enhanced layout."""
        x_col = config['x_axis']
        y_col = config['y_axis']
        
        fig = px.scatter(df, x=x_col, y=y_col, title=config.get('title', f'{y_col} vs {x_col}'))
        
        # FIXED: Enhanced layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )
        
        # Add trend line if enough data points
        if len(df) > 10:
            fig.add_trace(go.Scatter(
                x=df[x_col], 
                y=np.poly1d(np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1))(df[x_col]),
                mode='lines',
                name='Trend',
                line={'dash': 'dash', 'color': 'red'}
            ))
        
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create histogram with enhanced layout."""
        x_col = config['x_axis']
        
        fig = px.histogram(df, x=x_col, title=config.get('title', f'Distribution of {x_col}'))
        
        # FIXED: Enhanced layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title='Frequency',
            height=500,
            bargap=0.1
        )
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create box plot with enhanced layout."""
        x_col = config.get('x_axis')
        y_col = config.get('y_axis', config.get('x_axis'))
        
        if x_col and x_col != y_col and x_col in df.columns and y_col in df.columns:
            fig = px.box(df, x=x_col, y=y_col, title=config.get('title', f'{y_col} by {x_col}'))
        else:
            fig = px.box(df, y=y_col, title=config.get('title', f'Distribution of {y_col}'))
        
        # FIXED: Enhanced layout
        fig.update_layout(
            height=500
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pie chart with enhanced layout."""
        x_col = config['x_axis']
        
        counts = df[x_col].value_counts()
        
        # FIXED: Limit categories to top 10 for better readability
        if len(counts) > 10:
            top_counts = counts.head(10)
            other_sum = counts.tail(len(counts) - 10).sum()
            if other_sum > 0:
                top_counts['Others'] = other_sum
            counts = top_counts
        
        fig = px.pie(values=counts.values, names=counts.index,
                    title=config.get('title', f'Distribution of {x_col}'))
        
        # FIXED: Enhanced layout for pie charts
        fig.update_layout(
            height=500,
            showlegend=True,
            legend={'orientation': 'v', 'x': 1.05, 'xanchor': 'left'}
        )
        
        # Better hover info
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap with enhanced layout."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns")
        
        correlation = numeric_df.corr()
        
        fig = px.imshow(correlation, 
                       title=config.get('title', 'Correlation Heatmap'),
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        
        # FIXED: Enhanced layout for heatmap
        fig.update_layout(
            height=500,
            width=600
        )
        
        return fig
    
    def _generate_basic_insights(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
        """Generate basic insights about the created chart."""
        insights = []
        chart_type = config.get('chart_type', '').lower()
        x_col = config.get('x_axis')
        y_col = config.get('y_axis')
        
        try:
            insights.append(f"Chart displays {len(df):,} data points")
            
            if chart_type == 'histogram' and x_col:
                data = df[x_col].dropna()
                if len(data) > 0:
                    insights.append(f"Data ranges from {data.min():.2f} to {data.max():.2f}")
                    insights.append(f"Average value: {data.mean():.2f}")
                    
            elif chart_type == 'scatter' and x_col and y_col:
                if x_col in df.columns and y_col in df.columns:
                    correlation = df[x_col].corr(df[y_col])
                    if abs(correlation) > 0.7:
                        insights.append(f"Strong correlation ({correlation:.3f}) between variables")
                    elif abs(correlation) > 0.3:
                        insights.append(f"Moderate correlation ({correlation:.3f}) between variables")
                    else:
                        insights.append(f"Weak correlation ({correlation:.3f}) between variables")
                        
            elif chart_type == 'pie' and x_col:
                top_category = df[x_col].value_counts().index[0]
                count = df[x_col].value_counts().iloc[0]
                percentage = (count / len(df)) * 100
                insights.append(f"Most frequent category: {top_category} ({percentage:.1f}%)")
                
            elif chart_type == 'bar' and x_col:
                unique_categories = df[x_col].nunique()
                insights.append(f"Comparing {unique_categories} different categories")
                
        except Exception as e:
            insights.append("Chart created successfully")
        
        return insights
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="System Architecture",
    page_icon="🏗️",
    layout="wide"
)

def create_architecture_diagram():
    """Create a simple, readable layered architecture diagram"""
    fig = go.Figure()
    
    # Layer definitions with clear colors
    layers = [
        {'name': 'Frontend Layer', 'y': 4.5, 'color': '#2980b9', 'bg_color': '#ebf3fd'},
        {'name': 'Processing Layer', 'y': 3.5, 'color': '#c0392b', 'bg_color': '#fadbd8'},
        {'name': 'ML Models Layer', 'y': 2.5, 'color': '#27ae60', 'bg_color': '#d5f4e6'},
        {'name': 'Data Layer', 'y': 1.5, 'color': '#8e44ad', 'bg_color': '#f4ecf7'}
    ]
    
    # Add layer backgrounds with better contrast
    for layer in layers:
        fig.add_shape(
            type="rect",
            x0=1, x1=7,
            y0=layer['y']-0.35, y1=layer['y']+0.35,
            fillcolor=layer['bg_color'],
            opacity=0.8,
            line=dict(color=layer['color'], width=3)
        )
        
        # Add layer title on the left
        fig.add_annotation(
            x=0.5, y=layer['y'],
            text=f"<b>{layer['name']}</b>",
            showarrow=False,
            font=dict(size=16, color=layer['color']),
            textangle=0,
            xanchor='center'
        )
    
    # Define components with better spacing and readability
    components = [
        # Frontend Layer
        {'name': 'Dashboard<br>Interface', 'x': 2.5, 'y': 4.5, 'color': '#2980b9'},
        {'name': 'Interactive<br>Charts', 'x': 4, 'y': 4.5, 'color': '#2980b9'},
        {'name': 'Model<br>Comparison', 'x': 5.5, 'y': 4.5, 'color': '#2980b9'},
        
        # Processing Layer
        {'name': 'Data<br>Pipeline', 'x': 2.5, 'y': 3.5, 'color': '#c0392b'},
        {'name': 'Model<br>Manager', 'x': 4, 'y': 3.5, 'color': '#c0392b'},
        {'name': 'Performance<br>Ranking', 'x': 5.5, 'y': 3.5, 'color': '#c0392b'},
        
        # Models Layer
        {'name': 'N-BEATSx<br>Model', 'x': 2.5, 'y': 2.5, 'color': '#27ae60'},
        {'name': 'Interpretable<br>N-BEATS', 'x': 4, 'y': 2.5, 'color': '#27ae60'},
        {'name': 'Fast Directional<br>Forecaster', 'x': 5.5, 'y': 2.5, 'color': '#27ae60'},
        
        # Data Layer
        {'name': 'Government<br>API', 'x': 2.5, 'y': 1.5, 'color': '#8e44ad'},
        {'name': 'CSV Data<br>Storage', 'x': 4, 'y': 1.5, 'color': '#8e44ad'},
        {'name': 'Update<br>Scheduler', 'x': 5.5, 'y': 1.5, 'color': '#8e44ad'},
    ]
    
    # Add component boxes with better text contrast
    for comp in components:
        # Add white background box for better text readability
        fig.add_shape(
            type="rect",
            x0=comp['x']-0.5, x1=comp['x']+0.5,
            y0=comp['y']-0.2, y1=comp['y']+0.2,
            fillcolor='white',
            opacity=0.95,
            line=dict(color=comp['color'], width=3)
        )
        
        # Add component text with dark color for readability
        fig.add_annotation(
            x=comp['x'], y=comp['y'],
            text=f"<b>{comp['name']}</b>",
            showarrow=False,
            font=dict(size=12, color='#2c3e50'),
        )
    
    # Simplified data flow with numbered steps and clear direction
    flow_steps = [
        {'text': '1', 'x': 1.2, 'y': 4, 'desc': 'User Input'},
        {'text': '2', 'x': 3.2, 'y': 4, 'desc': 'Data Processing'},
        {'text': '3', 'x': 4, 'y': 3, 'desc': 'Model Training'},
        {'text': '4', 'x': 4, 'y': 2, 'desc': 'Data Access'},
        {'text': '5', 'x': 6.2, 'y': 4, 'desc': 'Results Display'}
    ]
    
    for step in flow_steps:
        # Add numbered circles for flow
        fig.add_shape(
            type="circle",
            x0=step['x']-0.15, x1=step['x']+0.15,
            y0=step['y']-0.1, y1=step['y']+0.1,
            fillcolor='#34495e',
            line=dict(color='#34495e', width=2)
        )
        
        fig.add_annotation(
            x=step['x'], y=step['y'],
            text=f"<b>{step['text']}</b>",
            showarrow=False,
            font=dict(size=12, color='white')
        )
        
        # Add step description
        fig.add_annotation(
            x=step['x'], y=step['y']-0.3,
            text=step['desc'],
            showarrow=False,
            font=dict(size=10, color='#7f8c8d')
        )
    
    fig.update_layout(
        title="<b>COE Prediction System Architecture</b><br><span style='font-size:14px'>Simple layered design with numbered data flow</span>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 7.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1, 5.2]),
        plot_bgcolor='white',
        height=650,
        margin=dict(l=120, r=20, t=80, b=20)
    )
    
    return fig

def create_data_flow_diagram():
    """Create a clean horizontal data flow visualization"""
    fig = go.Figure()
    
    # Data flow stages with better spacing
    stages = [
        {'name': 'Raw Data\nIngestion', 'x': 1, 'color': '#e74c3c', 'icon': '📥'},
        {'name': 'Data\nCleaning', 'x': 2.5, 'color': '#f39c12', 'icon': '🧹'},
        {'name': 'Feature\nEngineering', 'x': 4, 'color': '#f1c40f', 'icon': '⚙️'},
        {'name': 'Model\nTraining', 'x': 5.5, 'color': '#27ae60', 'icon': '🧠'},
        {'name': 'Prediction\nGeneration', 'x': 7, 'color': '#3498db', 'icon': '🔮'},
        {'name': 'Dashboard\nDisplay', 'x': 8.5, 'color': '#9b59b6', 'icon': '📊'}
    ]
    
    # Add stage boxes with icons
    for stage in stages:
        # Add rounded rectangle background
        fig.add_shape(
            type="rect",
            x0=stage['x']-0.4, x1=stage['x']+0.4,
            y0=2.7, y1=3.3,
            fillcolor=stage['color'],
            opacity=0.8,
            line=dict(color=stage['color'], width=2),
            layer="below"
        )
        
        # Add icon at top
        fig.add_annotation(
            x=stage['x'], y=3.45,
            text=stage['icon'],
            showarrow=False,
            font=dict(size=20)
        )
        
        # Add stage name
        fig.add_annotation(
            x=stage['x'], y=3,
            text=f"<b>{stage['name']}</b>",
            showarrow=False,
            font=dict(size=10, color='white')
        )
    
    # Add flow arrows between stages
    for i in range(len(stages) - 1):
        start_x = stages[i]['x'] + 0.4
        end_x = stages[i + 1]['x'] - 0.4
        
        fig.add_annotation(
            x=end_x, y=3,
            ax=start_x, ay=3,
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2, arrowsize=1.5, arrowwidth=3,
            arrowcolor='#34495e'
        )
    
    # Add process descriptions below
    descriptions = [
        "Singapore Gov API\n2,600 records",
        "Remove duplicates\nHandle missing values", 
        "Create indicators\nTime features",
        "3 ML models\nValidation testing",
        "Price forecasts\nDirectional signals",
        "Interactive charts\nReal-time updates"
    ]
    
    for i, desc in enumerate(descriptions):
        fig.add_annotation(
            x=stages[i]['x'], y=2.4,
            text=desc,
            showarrow=False,
            font=dict(size=9, color='#7f8c8d'),
            align='center'
        )
    
    fig.update_layout(
        title="<b>Data Processing Pipeline - End-to-End Flow</b>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.3, 9.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[2.2, 3.7]),
        plot_bgcolor='white',
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    """Main function for the System Architecture page"""
    st.title("🏗️ System Architecture Documentation")
    st.markdown("### Comprehensive Technical Architecture of COE Prediction Platform")
    
    # Introduction
    st.markdown("""
    ---
    This page provides detailed technical documentation of the COE prediction system architecture,
    covering all components from data ingestion to model deployment and user interface.
    """)
    
    # Architecture Overview
    st.header("📋 Architecture Overview")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig_arch = create_architecture_diagram()
        st.plotly_chart(fig_arch, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Design Principles")
        st.markdown("""
        **Modularity**: Separate concerns with distinct layers
        
        **Scalability**: Designed for easy model addition/removal
        
        **Maintainability**: Clear separation of data, models, and UI
        
        **Performance**: Optimized for real-time predictions
        
        **Reliability**: Robust error handling and fallbacks
        
        **Interpretability**: Transparent model operations
        """)
    
    # Component Details
    st.header("🔧 Component Details")
    
    # Create tabs for different layers
    frontend_tab, processing_tab, models_tab, data_tab = st.tabs([
        "🖥️ Frontend Layer", 
        "⚙️ Processing Layer", 
        "🧠 Models Layer", 
        "💾 Data Layer"
    ])
    
    with frontend_tab:
        st.subheader("Streamlit Frontend Architecture")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **🎨 User Interface Components:**
            
            **Main Dashboard (`app.py`)**
            - Dynamic model ranking system
            - Real-time prediction interface
            - Interactive charts and visualizations
            - Model performance comparison
            - Automated data update triggers
            
            **Testing & Validation Page**
            - Comprehensive model evaluation
            - Performance metrics visualization
            - Confusion matrices and ROC curves
            - Educational explanations
            
            **System Architecture Page** *(Current)*
            - Technical documentation
            - Component relationship diagrams
            - Data flow visualization
            """)
        
        with col2:
            st.markdown("""
            **🔧 Technical Specifications:**
            
            **Framework**: Streamlit 1.28+
            **Port**: 5000 (production-ready)
            **Caching**: @st.cache_data for performance
            **Session State**: Model persistence
            **Multi-page**: Native Streamlit pages/
            
            **Key Features:**
            - Responsive design
            - Real-time updates
            - Interactive plotting with Plotly
            - Professional styling
            - Mobile-friendly layout
            - Error handling and user feedback
            """)
        
        st.subheader("📱 User Experience Flow")
        fig_flow = create_data_flow_diagram()
        st.plotly_chart(fig_flow, use_container_width=True)
        
        st.success("""
        **Simple User Journey (5 Easy Steps):**
        
        **Step 1**: Load dashboard → See professional loading screen
        
        **Step 2**: Data updates → System checks for latest COE results
        
        **Step 3**: Choose model → Select from top-performing prediction models
        
        **Step 4**: Get predictions → View 6-cycle price forecasts with charts
        
        **Step 5**: Explore details → Check model performance and validation results
        """)
    
    with processing_tab:
        st.subheader("Data Processing Pipeline")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **🔄 Core Processing Components:**
            
            **Data Updater (`utils/data_updater.py`)**
            - Singapore Government API integration
            - Automated data freshness checks
            - CSV file management and backup
            - Error handling and retry logic
            - Data validation and quality checks
            
            **COE Scheduler (`utils/coe_scheduler.py`)**
            - Automated bidding date extraction
            - Web scraping from official sources
            - Background task scheduling
            - Real-time data monitoring
            """)
        
        with col2:
            st.markdown("""
            **⚡ Processing Features:**
            
            **Performance Optimization:**
            - Efficient data loading with pandas
            - Memory management for large datasets
            - Caching strategies for repeated operations
            - Parallel processing where applicable
            
            **Data Quality Assurance:**
            - Missing value detection and handling
            - Outlier identification and treatment
            - Data type validation and conversion
            - Historical consistency checks
            """)
        
        st.subheader("🔧 Processing Workflow")
        
        process_col1, process_col2, process_col3 = st.columns(3)
        
        with process_col1:
            st.markdown("""
            **📥 Data Ingestion**
            
            1. Check last update timestamp
            2. Query Singapore Gov API
            3. Validate response format
            4. Parse JSON to DataFrame
            5. Apply data type conversions
            """)
        
        with process_col2:
            st.markdown("""
            **🧹 Data Cleaning**
            
            1. Remove duplicate records
            2. Handle missing values
            3. Standardize category names
            4. Convert price strings to numeric
            5. Extract date components
            """)
        
        with process_col3:
            st.markdown("""
            **🔄 Data Integration**
            
            1. Merge with existing dataset
            2. Sort by bidding exercise
            3. Create backup copies
            4. Update model rankings
            5. Trigger model retraining
            """)
    
    with models_tab:
        st.subheader("Machine Learning Models Architecture")
        
        # Model comparison table
        model_specs = {
            'Feature': [
                'Algorithm Type',
                'Training Time',
                'Memory Usage',
                'Inference Speed',
                'Interpretability',
                'Accuracy',
                'Best Use Case'
            ],
            'Fast Directional Forecaster': [
                'Exponential Smoothing + Momentum',
                '< 1 minute',
                '< 50MB',
                '< 100ms',
                'High (momentum indicators)',
                '88.9%',
                'Real-time trading'
            ],
            'Interpretable N-BEATS': [
                'Decomposition + ML Ensemble',
                '2-3 minutes',
                '< 200MB',
                '< 500ms',
                'Very High (mathematical)',
                '90.6%',
                'Regulatory compliance'
            ],
            'N-BEATSx': [
                'Neural Basis Expansion',
                '5-10 minutes',
                '< 500MB',
                '< 1 second',
                'Medium (feature importance)',
                '92.7%',
                'Maximum accuracy'
            ]
        }
        
        df_specs = pd.DataFrame(model_specs)
        st.dataframe(df_specs, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🧠 Model Implementation Details")
            st.markdown("""
            **Fast Directional Forecaster:**
            - Exponential smoothing for trend detection
            - Technical indicators (RSI, momentum, volatility)
            - Statistical validation with walk-forward testing
            - Binary direction classification
            
            **Interpretable N-BEATS:**
            - Time series decomposition (trend + seasonal + residual)
            - Polynomial trend fitting (degree 2)
            - Fourier analysis for seasonality
            - Ensemble of specialized components
            
            **N-BEATSx:**
            - Neural basis expansion architecture
            - Exogenous variable integration
            - Advanced feature engineering
            - Deep learning optimization
            """)
        
        with col2:
            st.subheader("⚖️ Model Selection Strategy")
            st.markdown("""
            **Dynamic Ranking System:**
            - Performance-based automatic ordering
            - 70% price accuracy + 30% directional accuracy
            - Rolling 6-cycle validation window
            - Real-time performance monitoring
            
            **Ensemble Considerations:**
            - Each model captures different patterns
            - Complementary strengths and weaknesses
            - Potential for voting or weighted averaging
            - Risk diversification across approaches
            """)
    
    with data_tab:
        st.subheader("Data Layer Architecture")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **📊 Data Sources:**
            
            **Primary Source: Singapore Government API**
            - Official COE bidding results
            - Real-time updates after each exercise
            - Historical data from 2002-2025
            - 2,574 records across all categories
            
            **Data Schema:**
            - `month`: Bidding period (YYYY-MM format)
            - `bidding_no`: Exercise number within month
            - `vehicle_class`: Category (A, B, C, D, E, M)
            - `quota`: Available certificates
            - `bids_success`: Successful bids count
            - `bids_received`: Total bids received
            - `premium`: Winning bid price (SGD)
            """)
        
        with col2:
            st.markdown("""
            **💾 Storage Strategy:**
            
            **File-based Storage:**
            - CSV format for portability
            - Automatic backup creation
            - Version control integration
            - Fast pandas operations
            
            **Data Management:**
            - Incremental updates only
            - Duplicate detection and removal
            - Data integrity validation
            - Automated quality checks
            - Error logging and recovery
            """)
        
        st.subheader("🔐 Data Security and Quality")
        
        security_col1, security_col2, security_col3 = st.columns(3)
        
        with security_col1:
            st.markdown("""
            **🛡️ Security Measures**
            
            - Read-only API access
            - No personal data collection
            - Public data sources only
            - Secure HTTP connections
            - Input validation and sanitization
            """)
        
        with security_col2:
            st.markdown("""
            **✅ Quality Assurance**
            
            - Automated data validation
            - Outlier detection algorithms
            - Cross-reference with multiple sources
            - Historical consistency checks
            - Real-time monitoring alerts
            """)
        
        with security_col3:
            st.markdown("""
            **📈 Performance Optimization**
            
            - Efficient pandas operations
            - Memory-mapped file access
            - Lazy loading strategies
            - Caching frequently accessed data
            - Parallel processing capabilities
            """)
    
    # Technical Specifications
    st.header("⚙️ Technical Specifications")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.subheader("🖥️ System Requirements")
        st.markdown("""
        **Minimum Specifications:**
        - Python 3.8+
        - 2GB RAM
        - 1GB storage
        - Internet connection
        
        **Recommended:**
        - Python 3.11+
        - 4GB RAM
        - 2GB storage
        - Stable broadband
        """)
    
    with tech_col2:
        st.subheader("📦 Dependencies")
        st.markdown("""
        **Core Libraries:**
        - streamlit >= 1.28
        - pandas >= 1.5
        - plotly >= 5.0
        - numpy >= 1.21
        - scikit-learn >= 1.0
        - requests >= 2.28
        """)
    
    with tech_col3:
        st.subheader("🚀 Deployment")
        st.markdown("""
        **Platforms:**
        - Replit (current)
        - Streamlit Cloud
        - Heroku
        - Docker containers
        
        **Features:**
        - Auto-scaling
        - Health checks
        - Error monitoring
        """)
    
    # Future Enhancements
    st.header("🔮 Future Architecture Enhancements")
    
    future_col1, future_col2 = st.columns([1, 1])
    
    with future_col1:
        st.subheader("📈 Planned Improvements")
        st.markdown("""
        **Database Integration:**
        - PostgreSQL for production data
        - Redis for caching layer
        - Time-series optimization
        
        **API Development:**
        - RESTful prediction endpoints
        - Authentication and rate limiting
        - API documentation with OpenAPI
        
        **Advanced Analytics:**
        - A/B testing framework
        - User behavior tracking
        - Performance dashboards
        """)
    
    with future_col2:
        st.subheader("🔧 Scalability Considerations")
        st.markdown("""
        **Microservices Architecture:**
        - Separate model serving services
        - Load balancing and auto-scaling
        - Container orchestration
        
        **Real-time Processing:**
        - Streaming data pipelines
        - Event-driven architecture
        - WebSocket connections
        
        **Machine Learning Operations:**
        - Model versioning and A/B testing
        - Automated retraining pipelines
        - Performance monitoring and alerts
        """)

if __name__ == "__main__":
    main()
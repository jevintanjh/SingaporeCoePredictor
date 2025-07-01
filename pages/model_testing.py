
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fast_directional_forecaster import FastDirectionalForecaster
from models.interpretable_nbeats_v2 import InterpretableNBEATS
from models.nbeatsx import NBEATSx
from utils.validation_dashboard import display_model_validation_dashboard

# Configure page
st.set_page_config(
    page_title="COE Model Testing Dashboard",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0;
}

.info-banner {
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    color: #374151;
}

.chart-container {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin: 1rem 0;
    border: 1px solid #e5e7eb;
}

/* Hide streamlit style */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the COE data"""
    try:
        # Try different file paths
        file_paths = [
            'data/COE_Clean_2002_2025.csv',
            '../data/COE_Clean_2002_2025.csv',
            'attached_assets/COEBiddingResultsPrices_1749430265007.csv',
            '../attached_assets/COEBiddingResultsPrices_1749430265007.csv'
        ]

        data = None
        for file_path in file_paths:
            try:
                data = pd.read_csv(file_path)
                break
            except:
                continue

        if data is None:
            st.error("Could not load data file. Please check file paths.")
            return None

        # Clean and process the data (same logic as main app)
        data = data.copy()

        # Handle different column names and formats
        if 'month' in data.columns and 'bidding_no' in data.columns:
            data['date'] = pd.to_datetime(data['month'] + '-01') + pd.to_timedelta((data['bidding_no'] - 1) * 15, unit='D')
        elif 'exercise' in data.columns:
            data['date'] = data['exercise'].apply(extract_date_info)
        else:
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                data['date'] = pd.to_datetime(data[date_cols[0]])
            else:
                data['date'] = pd.date_range(start='2002-01-01', periods=len(data), freq='2W')

        # Clean premium prices
        if 'premium' in data.columns:
            data['premium'] = data['premium'].apply(clean_price)
        elif any('price' in col.lower() for col in data.columns):
            price_col = [col for col in data.columns if 'price' in col.lower()][0]
            data['premium'] = data[price_col].apply(clean_price)

        # Standardize category names
        if 'vehicle_class' in data.columns:
            data['vehicle_class'] = data['vehicle_class'].apply(standardize_category)
        elif any('category' in col.lower() for col in data.columns):
            cat_col = [col for col in data.columns if 'category' in col.lower()][0]
            data['vehicle_class'] = data[cat_col].apply(standardize_category)

        # Remove rows with missing essential data
        data = data.dropna(subset=['date', 'premium', 'vehicle_class'])
        data = data.sort_values('date')

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_price(price_str):
    """Clean price string to numeric value"""
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str)
    price_str = price_str.replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(price_str)
    except:
        return np.nan

def extract_date_info(exercise):
    """Extract date information from exercise string"""
    try:
        if pd.isna(exercise):
            return pd.NaT
        exercise_str = str(exercise)
        import re
        date_match = re.search(r'(\d{4})[/-](\d{1,2})', exercise_str)
        if date_match:
            year, month = date_match.groups()
            day = 15 if '2' in exercise_str or 'second' in exercise_str.lower() else 1
            return pd.to_datetime(f"{year}-{month:0>2}-{day:0>2}")
        return pd.NaT
    except:
        return pd.NaT

def standardize_category(cat):
    """Standardize category names"""
    if pd.isna(cat):
        return None
    cat_str = str(cat).upper().strip()
    
    if cat_str == 'CATEGORY A' or cat_str == 'A':
        return 'Category A'
    elif cat_str == 'CATEGORY B' or cat_str == 'B':
        return 'Category B'
    elif cat_str == 'CATEGORY C' or cat_str == 'C':
        return 'Category C'
    elif cat_str == 'CATEGORY D' or cat_str == 'D':
        return 'Category D'
    elif cat_str == 'CATEGORY E' or cat_str == 'E':
        return 'Category E'
    elif 'CATEGORY A' in cat_str or ('A' in cat_str and ('CARS' in cat_str or 'SMALL' in cat_str)):
        return 'Category A'
    elif 'CATEGORY B' in cat_str or ('B' in cat_str and 'CATEGORY' in cat_str):
        return 'Category B'
    elif 'CATEGORY C' in cat_str or ('C' in cat_str and ('GOODS' in cat_str or 'LIGHT' in cat_str)):
        return 'Category C'
    elif 'CATEGORY D' in cat_str or ('D' in cat_str and 'CATEGORY' in cat_str):
        return 'Category D'
    elif 'CATEGORY E' in cat_str or ('E' in cat_str and ('OPEN' in cat_str or 'BIG' in cat_str)):
        return 'Category E'
    if cat_str.startswith('CATEGORY'):
        return cat_str.title()
    return None

def main():
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Model Testing & Validation Dashboard</h1>
        <p>Comprehensive analysis and comparison of all COE prediction models</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation back to main page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Main Dashboard", type="secondary"):
            st.switch_page("app.py")
    
    with col3:
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    data = load_and_process_data()
    
    if data is None:
        st.error("Unable to load COE data. Please check the data files.")
        return

    # Data info
    total_records = len(data)
    date_range = f"{data['date'].min().strftime('%d-%m-%Y')} to {data['date'].max().strftime('%d-%m-%Y')}"
    
    st.markdown(f"""
    <div class="info-banner">
        <strong>📊 Dataset:</strong> {total_records:,} historical COE bidding records ({date_range})<br>
        <small><strong>Analysis Date:</strong> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}</small>
    </div>
    """, unsafe_allow_html=True)

    # Display the validation dashboard
    try:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        display_model_validation_dashboard(data)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading validation dashboard: {str(e)}")
        st.info("Make sure all required model files are available and properly configured.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details"):
            st.code(str(e))

    # Additional insights section
    st.markdown("---")
    st.subheader("📋 Testing Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Validation Approach:**
        - Walk-forward validation with realistic time splits
        - Bootstrap sampling for stability assessment
        - Cross-validation across different market regimes
        - Performance tracking over multiple time horizons
        """)
    
    with col2:
        st.markdown("""
        **Key Metrics:**
        - **R² Score**: Variance explained by the model
        - **Direction Accuracy**: Correct prediction of price movement
        - **MAPE**: Mean Absolute Percentage Error
        - **RMSE/MAE**: Absolute prediction errors
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <p>Model testing results are updated automatically with each new COE exercise.<br>
        Performance rankings on the main dashboard reflect these comprehensive test results.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

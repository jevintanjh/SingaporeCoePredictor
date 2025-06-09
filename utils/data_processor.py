import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data processing utilities for COE bidding data
    """
    
    def __init__(self):
        self.required_columns = ['month', 'bidding_no', 'vehicle_class', 'quota', 
                               'bids_success', 'bids_received', 'premium']
    
    def load_data(self, file_path):
        """Load COE data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return data
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")
    
    def preprocess_data(self, data):
        """Preprocess the COE data"""
        df = data.copy()
        
        # Create proper date column
        df['date'] = pd.to_datetime(df['month'] + '-' + df['bidding_no'].astype(str).str.zfill(2) + '-01')
        
        # Adjust date for second bidding (add 15 days)
        df.loc[df['bidding_no'] == 2, 'date'] = df.loc[df['bidding_no'] == 2, 'date'] + pd.DateOffset(days=15)
        
        # Clean and convert data types
        df = self.clean_data_types(df)
        
        # Remove duplicates and sort
        df = df.drop_duplicates().sort_values(['date', 'vehicle_class']).reset_index(drop=True)
        
        # Validate data quality
        df = self.validate_data_quality(df)
        
        return df
    
    def clean_data_types(self, df):
        """Clean and convert data types"""
        # Convert numeric columns
        numeric_cols = ['quota', 'bids_success', 'bids_received', 'premium']
        
        for col in numeric_cols:
            # Remove any non-numeric characters except digits and decimal points
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle bidding_no
        df['bidding_no'] = pd.to_numeric(df['bidding_no'], errors='coerce')
        
        # Clean vehicle_class
        df['vehicle_class'] = df['vehicle_class'].astype(str).str.strip()
        
        return df
    
    def validate_data_quality(self, df):
        """Validate and clean data quality issues"""
        # Remove rows with missing critical data
        critical_cols = ['date', 'vehicle_class', 'premium', 'quota']
        df = df.dropna(subset=critical_cols)
        
        # Remove rows with zero or negative premiums
        df = df[df['premium'] > 0]
        
        # Remove rows with zero quotas
        df = df[df['quota'] > 0]
        
        # Ensure bids_success <= bids_received
        df.loc[df['bids_success'] > df['bids_received'], 'bids_success'] = df['bids_received']
        
        # Remove obvious outliers (premiums > $200,000)
        df = df[df['premium'] <= 200000]
        
        # Fill missing bids_received with bids_success if available
        df['bids_received'] = df['bids_received'].fillna(df['bids_success'])
        df['bids_success'] = df['bids_success'].fillna(df['bids_received'])
        
        return df
    
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Extract time components
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_year_end'] = (df['month_num'] >= 11).astype(int)
        df['is_mid_year'] = (df['month_num'].isin([6, 7])).astype(int)
        
        # Create cyclical features for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        
        return df
    
    def create_lag_features(self, df, category=None, lag_periods=[1, 2, 3]):
        """Create lagged features for time series analysis"""
        df = df.copy()
        
        if category:
            # Process single category
            mask = df['vehicle_class'] == category
            category_data = df[mask].sort_values('date')
            
            for lag in lag_periods:
                df.loc[mask, f'premium_lag_{lag}'] = category_data['premium'].shift(lag)
                df.loc[mask, f'quota_lag_{lag}'] = category_data['quota'].shift(lag)
                df.loc[mask, f'bids_received_lag_{lag}'] = category_data['bids_received'].shift(lag)
        else:
            # Process all categories
            for cat in df['vehicle_class'].unique():
                mask = df['vehicle_class'] == cat
                category_data = df[mask].sort_values('date')
                
                for lag in lag_periods:
                    df.loc[mask, f'premium_lag_{lag}'] = category_data['premium'].shift(lag)
                    df.loc[mask, f'quota_lag_{lag}'] = category_data['quota'].shift(lag)
                    df.loc[mask, f'bids_received_lag_{lag}'] = category_data['bids_received'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, category=None, windows=[3, 6, 12]):
        """Create rolling window features"""
        df = df.copy()
        
        if category:
            # Process single category
            mask = df['vehicle_class'] == category
            category_data = df[mask].sort_values('date')
            
            for window in windows:
                df.loc[mask, f'premium_ma_{window}'] = category_data['premium'].rolling(window=window).mean()
                df.loc[mask, f'premium_std_{window}'] = category_data['premium'].rolling(window=window).std()
                df.loc[mask, f'quota_ma_{window}'] = category_data['quota'].rolling(window=window).mean()
        else:
            # Process all categories
            for cat in df['vehicle_class'].unique():
                mask = df['vehicle_class'] == cat
                category_data = df[mask].sort_values('date')
                
                for window in windows:
                    df.loc[mask, f'premium_ma_{window}'] = category_data['premium'].rolling(window=window).mean()
                    df.loc[mask, f'premium_std_{window}'] = category_data['premium'].rolling(window=window).std()
                    df.loc[mask, f'quota_ma_{window}'] = category_data['quota'].rolling(window=window).mean()
        
        return df
    
    def create_ratio_features(self, df):
        """Create ratio-based features"""
        df = df.copy()
        
        # Demand-supply ratios
        df['bid_quota_ratio'] = df['bids_received'] / df['quota']
        df['success_rate'] = df['bids_success'] / df['bids_received']
        df['oversubscription_ratio'] = (df['bids_received'] - df['quota']) / df['quota']
        
        # Handle division by zero
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def get_category_summary(self, df):
        """Get summary statistics for each category"""
        summary_stats = []
        
        for category in df['vehicle_class'].unique():
            cat_data = df[df['vehicle_class'] == category]
            
            stats = {
                'category': category,
                'total_records': len(cat_data),
                'date_range': f"{cat_data['date'].min().strftime('%Y-%m')} to {cat_data['date'].max().strftime('%Y-%m')}",
                'avg_premium': cat_data['premium'].mean(),
                'median_premium': cat_data['premium'].median(),
                'std_premium': cat_data['premium'].std(),
                'min_premium': cat_data['premium'].min(),
                'max_premium': cat_data['premium'].max(),
                'avg_quota': cat_data['quota'].mean(),
                'avg_bids_received': cat_data['bids_received'].mean(),
                'avg_success_rate': (cat_data['bids_success'] / cat_data['bids_received']).mean()
            }
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers in premium data"""
        outliers = []
        
        for category in df['vehicle_class'].unique():
            cat_data = df[df['vehicle_class'] == category]
            premiums = cat_data['premium']
            
            if method == 'iqr':
                Q1 = premiums.quantile(0.25)
                Q3 = premiums.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (premiums < lower_bound) | (premiums > upper_bound)
                outlier_indices = cat_data[outlier_mask].index.tolist()
                outliers.extend(outlier_indices)
        
        return outliers
    
    def get_data_quality_report(self, df):
        """Generate a data quality report"""
        report = {
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'categories': df['vehicle_class'].nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Category-wise record counts
        report['records_per_category'] = df['vehicle_class'].value_counts().to_dict()
        
        # Outliers
        outliers = self.detect_outliers(df)
        report['outlier_count'] = len(outliers)
        
        return report

import pandas as pd
import numpy as np
import re
from datetime import datetime

def clean_price_column(price_str):
    """Clean price column by removing $ and commas"""
    if pd.isna(price_str) or price_str == '':
        return 0
    
    # Convert to string and clean
    price_str = str(price_str)
    # Remove $ sign and commas
    cleaned = re.sub(r'[\$,"]', '', price_str)
    
    try:
        return int(cleaned)
    except:
        return 0

def clean_numeric_column(value):
    """Clean numeric columns by removing commas and quotes"""
    if pd.isna(value) or value == '':
        return 0
    
    # Convert to string and clean
    value_str = str(value)
    # Remove commas and quotes
    cleaned = re.sub(r'[,"]', '', value_str)
    
    try:
        return int(cleaned)
    except:
        return 0

def extract_month_year_bidding(bidding_exercise):
    """Extract month, year, and bidding number from exercise name"""
    # Example: "June 2025 First Open Bidding Exercise" -> 2025, 6, 1
    # Example: "December 2002 Second Open Bidding Exercise" -> 2002, 12, 2
    
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Extract components
    parts = bidding_exercise.lower().split()
    
    # Find month
    month = None
    for part in parts:
        if part in months:
            month = months[part]
            break
    
    # Find year
    year = None
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break
    
    # Find bidding number (first/second)
    bidding_no = 1 if 'first' in bidding_exercise.lower() else 2
    
    return year, month, bidding_no

def standardize_category_name(category):
    """Standardize category names to match existing format"""
    category = str(category).strip()
    
    if 'Cat A' in category or 'up to 1600cc' in category:
        return 'Category A'
    elif 'Cat B' in category or 'above 1600cc' in category:
        return 'Category B'
    elif 'Cat C' in category or 'Goods vehicles' in category or 'buses' in category:
        return 'Category C'
    elif 'Cat D' in category or 'Motorcycles' in category:
        return 'Category D'
    elif 'Cat E' in category or 'Open' in category:
        return 'Category E'
    else:
        return category

def process_new_dataset():
    """Process the new comprehensive COE dataset"""
    
    # Read the new dataset
    df_new = pd.read_csv('../attached_assets/Results of COE Bidding Exercise - Results_1749485110740.csv')
    
    print(f"Loaded new dataset with {len(df_new)} records")
    print(f"Date range: {df_new['Year'].min()} to {df_new['Year'].max()}")
    
    # Extract month, year, bidding_no from exercise name
    df_new[['year_extracted', 'month_extracted', 'bidding_no']] = df_new['Bidding Exercise'].apply(
        lambda x: pd.Series(extract_month_year_bidding(x))
    )
    
    # Clean and standardize columns
    df_new['quota_clean'] = df_new['Quota'].apply(clean_numeric_column)
    df_new['premium_clean'] = df_new['Quota Premium'].apply(clean_price_column)
    df_new['bids_received_clean'] = df_new['Total Bids Received'].apply(clean_numeric_column)
    df_new['bids_success_clean'] = df_new['Number of Successful Bids'].apply(clean_numeric_column)
    df_new['vehicle_class_clean'] = df_new['Category'].apply(standardize_category_name)
    
    # Create the standardized format to match existing dataset
    df_processed = pd.DataFrame({
        'month': df_new['year_extracted'].astype(str) + '-' + df_new['month_extracted'].astype(str).str.zfill(2),
        'bidding_no': df_new['bidding_no'],
        'vehicle_class': df_new['vehicle_class_clean'],
        'quota': df_new['quota_clean'],
        'bids_success': df_new['bids_success_clean'],
        'bids_received': df_new['bids_received_clean'],
        'premium': df_new['premium_clean']
    })
    
    # Remove invalid records
    df_processed = df_processed[
        (df_processed['quota'] > 0) & 
        (df_processed['premium'] > 0) &
        (df_processed['vehicle_class'].isin(['Category A', 'Category B', 'Category C', 'Category D', 'Category E']))
    ]
    
    # Sort by date (descending to match existing format)
    df_processed['sort_date'] = pd.to_datetime(df_processed['month'])
    df_processed = df_processed.sort_values(['sort_date', 'bidding_no'], ascending=[False, False])
    df_processed = df_processed.drop('sort_date', axis=1)
    
    print(f"Processed dataset: {len(df_processed)} valid records")
    print(f"Categories: {df_processed['vehicle_class'].value_counts().to_dict()}")
    print(f"Date range: {df_processed['month'].min()} to {df_processed['month'].max()}")
    
    return df_processed

def combine_datasets():
    """Combine new dataset with existing dataset"""
    
    # Process new dataset
    df_new = process_new_dataset()
    
    # Read existing dataset
    df_existing = pd.read_csv('data/COEBiddingResultsPrices_1749430265007.csv')
    
    print(f"Existing dataset: {len(df_existing)} records")
    print(f"Existing date range: {df_existing['month'].min()} to {df_existing['month'].max()}")
    
    # Find overlap and new data
    existing_dates = set(df_existing['month'].unique())
    new_dates = set(df_new['month'].unique())
    
    overlapping_dates = existing_dates.intersection(new_dates)
    truly_new_dates = new_dates - existing_dates
    
    print(f"Overlapping months: {len(overlapping_dates)}")
    print(f"New months to add: {len(truly_new_dates)}")
    
    # Filter to only truly new data (avoid duplicates)
    df_new_unique = df_new[df_new['month'].isin(truly_new_dates)]
    
    # Combine datasets
    df_combined = pd.concat([df_existing, df_new_unique], ignore_index=True)
    
    # Sort by date (descending)
    df_combined['sort_date'] = pd.to_datetime(df_combined['month'])
    df_combined = df_combined.sort_values(['sort_date', 'bidding_no'], ascending=[False, False])
    df_combined = df_combined.drop('sort_date', axis=1)
    
    print(f"Combined dataset: {len(df_combined)} total records")
    print(f"Final date range: {df_combined['month'].min()} to {df_combined['month'].max()}")
    
    # Save combined dataset
    df_combined.to_csv('data/COE_Combined_2002_2025.csv', index=False)
    
    return df_combined

def validate_combined_data():
    """Validate the combined dataset"""
    
    df = pd.read_csv('data/COE_Combined_2002_2025.csv')
    
    print(f"=== VALIDATION REPORT ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"Categories: {sorted(df['vehicle_class'].unique())}")
    print(f"Records per category:")
    for category in sorted(df['vehicle_class'].unique()):
        count = len(df[df['vehicle_class'] == category])
        print(f"  {category}: {count} records")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Check price ranges
    print(f"\nPrice statistics:")
    print(df['premium'].describe())
    
    # Check recent data
    print(f"\nMost recent entries:")
    print(df.head(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])
    
    # Check oldest data
    print(f"\nOldest entries:")
    print(df.tail(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])
    
    return df

if __name__ == "__main__":
    # Process and combine datasets
    df_combined = combine_datasets()
    
    # Validate results
    validate_combined_data()
    
    print(f"\n✅ Dataset combination completed successfully!")
    print(f"📁 Saved as: data/COE_Combined_2002_2025.csv")
import pandas as pd
import re

# Read new comprehensive dataset
print("Loading comprehensive COE dataset (2002-2025)...")
df_new = pd.read_csv('attached_assets/Results of COE Bidding Exercise - Results_1749485110740.csv')
print(f"Loaded {len(df_new)} records")

# Clean functions
def clean_price(price_str):
    if pd.isna(price_str): 
        return 0
    cleaned = re.sub(r'[\$,"]', '', str(price_str))
    try:
        return int(cleaned)
    except:
        return 0

def clean_numeric(val):
    if pd.isna(val):
        return 0
    cleaned = re.sub(r'[,"]', '', str(val))
    try:
        return int(cleaned)
    except:
        return 0

def extract_date_info(exercise):
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    parts = exercise.lower().split()
    
    # Find month
    month = 1
    for part in parts:
        if part in months:
            month = months[part]
            break
    
    # Find year
    year = 2000
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
            break
    
    # Find bidding number
    bidding = 1 if 'first' in exercise.lower() else 2
    
    return f"{year}-{month:02d}", bidding

def standardize_category(cat):
    cat_str = str(cat)
    if 'Cat A' in cat_str or '1600cc' in cat_str:
        return 'Category A'
    elif 'Cat B' in cat_str or 'above 1600cc' in cat_str:
        return 'Category B'
    elif 'Cat C' in cat_str or 'Goods' in cat_str or 'buses' in cat_str:
        return 'Category C'
    elif 'Cat D' in cat_str or 'Motorcycles' in cat_str:
        return 'Category D'
    elif 'Cat E' in cat_str or 'Open' in cat_str:
        return 'Category E'
    return cat_str

# Process the new dataset
print("Processing new dataset...")

# Extract date information
date_info = df_new['Bidding Exercise'].apply(extract_date_info)
df_new['month'] = [info[0] for info in date_info]
df_new['bidding_no'] = [info[1] for info in date_info]

# Clean and standardize
df_processed = pd.DataFrame({
    'month': df_new['month'],
    'bidding_no': df_new['bidding_no'],
    'vehicle_class': df_new['Category'].apply(standardize_category),
    'quota': df_new['Quota'].apply(clean_numeric),
    'bids_success': df_new['Number of Successful Bids'].apply(clean_numeric),
    'bids_received': df_new['Total Bids Received'].apply(clean_numeric),
    'premium': df_new['Quota Premium'].apply(clean_price)
})

# Filter valid records
valid_categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
df_processed = df_processed[
    (df_processed['quota'] > 0) & 
    (df_processed['premium'] > 0) &
    (df_processed['vehicle_class'].isin(valid_categories))
]

print(f"Processed {len(df_processed)} valid records")

# Read existing dataset
print("Reading existing dataset...")
df_existing = pd.read_csv('data/COEBiddingResultsPrices_1749430265007.csv')
print(f"Existing dataset: {len(df_existing)} records")

# Find new dates only (avoid duplicates)
existing_dates = set(df_existing['month'].unique())
new_dates = set(df_processed['month'].unique())
new_only_dates = list(new_dates - existing_dates)

# Filter to only new data
df_new_data = df_processed[df_processed['month'].isin(new_only_dates)]

print(f"Found {len(df_new_data)} new records to add")

# Combine datasets
df_combined = pd.concat([df_existing, df_new_data], ignore_index=True)

# Sort by date (newest first)
df_combined['temp_date'] = pd.to_datetime(df_combined['month'])
df_combined = df_combined.sort_values(['temp_date', 'bidding_no'], ascending=[False, False])
df_combined = df_combined.drop('temp_date', axis=1)

# Save combined dataset
df_combined.to_csv('data/COE_Extended_2002_2025.csv', index=False)

print("\n=== FINAL DATASET SUMMARY ===")
print(f"Total records: {len(df_combined)}")
print(f"Date range: {df_combined['month'].min()} to {df_combined['month'].max()}")
print(f"New records added: {len(df_new_data)}")

print("\nRecords per category:")
for cat in sorted(df_combined['vehicle_class'].unique()):
    count = len(df_combined[df_combined['vehicle_class'] == cat])
    print(f"  {cat}: {count} records")

print("\nRecent entries:")
print(df_combined.head(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])

print("\nOldest entries:")
print(df_combined.tail(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])

print(f"\n✅ Extended dataset saved as: data/COE_Extended_2002_2025.csv")
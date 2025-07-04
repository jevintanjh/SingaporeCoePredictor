import pandas as pd

# Read the combined dataset
df = pd.read_csv('data/COE_Extended_2002_2025.csv')
print(f"Original dataset: {len(df)} records")

# Remove duplicates by keeping the first occurrence of each unique combination
df_clean = df.drop_duplicates(subset=['month', 'bidding_no', 'vehicle_class'], keep='first')
print(f"After removing duplicates: {len(df_clean)} records")
print(f"Removed {len(df) - len(df_clean)} duplicate records")

# Verify no duplicates remain
duplicates_check = df_clean.groupby(['month', 'bidding_no', 'vehicle_class']).size()
remaining_duplicates = duplicates_check[duplicates_check > 1]
print(f"Remaining duplicates: {len(remaining_duplicates)}")

# Sort by date (newest first) and save
df_clean['temp_date'] = pd.to_datetime(df_clean['month'])
df_clean = df_clean.sort_values(['temp_date', 'bidding_no'], ascending=[False, False])
df_clean = df_clean.drop('temp_date', axis=1)

# Save cleaned dataset
df_clean.to_csv('data/COE_Clean_2002_2025.csv', index=False)

print(f"\n=== FINAL CLEAN DATASET ===")
print(f"Total records: {len(df_clean)}")
print(f"Date range: {df_clean['month'].min()} to {df_clean['month'].max()}")
print("Records per category:")
for cat in sorted(df_clean['vehicle_class'].unique()):
    count = len(df_clean[df_clean['vehicle_class'] == cat])
    print(f"  {cat}: {count} records")

# Sample recent and old data
print("\nMost recent entries:")
print(df_clean.head(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])

print("\nOldest entries:")
print(df_clean.tail(10)[['month', 'bidding_no', 'vehicle_class', 'premium']])

print(f"\n✅ Clean dataset saved as: data/COE_Clean_2002_2025.csv")
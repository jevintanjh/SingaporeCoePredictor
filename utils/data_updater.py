import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

class COEDataUpdater:
    """
    Automated COE data updater using Singapore Government Open Data API
    """
    
    def __init__(self):
        self.api_base_url = "https://data.gov.sg/api/action/datastore_search"
        # Try different potential dataset IDs for COE data
        self.dataset_ids = [
            "d_09c3388ad7e51af3a7dcc84eba52b68",  # Primary COE results dataset ID
            "09c3388a-d7e5-1af3-a7dc-c84eba52b68a",  # Alternative format
            "coe-bidding-results",  # Simplified ID
        ]
        self.data_file_path = "data/COE_Clean_2002_2025.csv"
        self.backup_file_path = "data/COE_Clean_backup.csv"
        
    def fetch_latest_coe_data(self, limit=1000):
        """
        Fetch latest COE data from Singapore Government API
        """
        print(f"Fetching COE data from government API...")
        
        for dataset_id in self.dataset_ids:
            try:
                # Construct API request
                params = {
                    'resource_id': dataset_id,
                    'limit': limit,
                    'sort': 'bidding_no desc'  # Get most recent data first
                }
                
                print(f"Trying dataset ID: {dataset_id}")
                response = requests.get(self.api_base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('success') and 'result' in data and 'records' in data['result']:
                        records = data['result']['records']
                        if records:  # Only proceed if we have actual records
                            print(f"Successfully fetched {len(records)} records from API using dataset ID: {dataset_id}")
                            return self.process_api_records(records)
                    else:
                        print(f"API response format unexpected for dataset {dataset_id}: {data}")
                        continue
                else:
                    print(f"API request failed for dataset {dataset_id} with status code: {response.status_code}")
                    if response.status_code == 404:
                        continue  # Try next dataset ID
                    elif response.status_code == 409:
                        print("Rate limiting detected, waiting before retry...")
                        import time
                        time.sleep(2)
                        continue
                    
            except Exception as e:
                print(f"Error fetching data with dataset {dataset_id}: {str(e)}")
                continue
        
        print("Failed to fetch data from all available dataset IDs")
        return None
    
    def process_api_records(self, records):
        """
        Process and clean API records into standardized format
        """
        try:
            processed_data = []
            
            for record in records:
                try:
                    # Extract and clean the data
                    processed_record = {
                        'date': self.parse_date(record.get('month', '')),
                        'vehicle_class': self.standardize_category(record.get('vehicle_class', '')),
                        'premium': self.clean_price(record.get('premium', 0)),
                        'quota': self.clean_numeric(record.get('quota', 0)),
                        'bids_received': self.clean_numeric(record.get('bids_received', 0)),
                        'bids_success': self.clean_numeric(record.get('bids_success', 0)),
                        'bidding_no': record.get('bidding_no', ''),
                        'exercise': record.get('month', '') + ' ' + record.get('bidding_no', '')
                    }
                    
                    # Only add valid records
                    if processed_record['date'] is not None and processed_record['premium'] > 0:
                        processed_data.append(processed_record)
                        
                except Exception as e:
                    print(f"Error processing record: {str(e)}")
                    continue
            
            if processed_data:
                df = pd.DataFrame(processed_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(['date', 'vehicle_class']).reset_index(drop=True)
                print(f"Processed {len(df)} valid records")
                return df
            else:
                print("No valid records processed")
                return None
                
        except Exception as e:
            print(f"Error processing API records: {str(e)}")
            return None
    
    def parse_date(self, month_str):
        """
        Parse date string from API format to datetime
        """
        try:
            # Expected format: "2025-01" or similar
            if len(month_str) >= 7:  # YYYY-MM format
                year_month = month_str[:7]
                # For COE, assume 1st of month for first exercise, 15th for second
                # This is a simplification - in practice, you'd need bidding_no to determine exact date
                date_obj = datetime.strptime(year_month + "-01", "%Y-%m-%d")
                return date_obj
            return None
        except:
            return None
    
    def standardize_category(self, category):
        """
        Standardize vehicle category names
        """
        category = str(category).strip().upper()
        
        if 'CAT A' in category or 'CATEGORY A' in category:
            return 'Category A'
        elif 'CAT B' in category or 'CATEGORY B' in category:
            return 'Category B'
        elif 'CAT C' in category or 'CATEGORY C' in category:
            return 'Category C'
        elif 'CAT D' in category or 'CATEGORY D' in category:
            return 'Category D'
        elif 'CAT E' in category or 'CATEGORY E' in category:
            return 'Category E'
        else:
            return category
    
    def clean_price(self, price_str):
        """
        Clean and convert price to numeric
        """
        try:
            if isinstance(price_str, (int, float)):
                return float(price_str)
            
            price_str = str(price_str).replace(',', '').replace('$', '').strip()
            return float(price_str) if price_str else 0
        except:
            return 0
    
    def clean_numeric(self, value):
        """
        Clean and convert numeric values
        """
        try:
            if isinstance(value, (int, float)):
                return int(value)
            
            value_str = str(value).replace(',', '').strip()
            return int(float(value_str)) if value_str else 0
        except:
            return 0
    
    def load_existing_data(self):
        """
        Load existing COE data from file
        """
        try:
            if os.path.exists(self.data_file_path):
                df = pd.read_csv(self.data_file_path)
                df['date'] = pd.to_datetime(df['date'])
                print(f"Loaded {len(df)} existing records")
                return df
            else:
                print("No existing data file found")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            return pd.DataFrame()
    
    def merge_and_deduplicate(self, existing_df, new_df):
        """
        Merge new data with existing data and remove duplicates
        """
        try:
            if new_df is None or len(new_df) == 0:
                print("No new data to merge")
                return existing_df
            
            if len(existing_df) == 0:
                print("No existing data, using new data only")
                return new_df
            
            # Combine datasets
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on date, vehicle_class, and exercise
            before_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'vehicle_class', 'exercise'], 
                keep='last'
            ).reset_index(drop=True)
            after_count = len(combined_df)
            
            print(f"Merged data: {before_count} total records, {after_count} after deduplication")
            print(f"Added {after_count - len(existing_df)} new records")
            
            # Sort by date
            combined_df = combined_df.sort_values(['date', 'vehicle_class']).reset_index(drop=True)
            
            return combined_df
            
        except Exception as e:
            print(f"Error merging data: {str(e)}")
            return existing_df
    
    def save_updated_data(self, df):
        """
        Save updated data to file with backup
        """
        try:
            # Create backup of existing file
            if os.path.exists(self.data_file_path):
                import shutil
                shutil.copy2(self.data_file_path, self.backup_file_path)
                print("Created backup of existing data")
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
            
            # Save updated data
            df.to_csv(self.data_file_path, index=False)
            print(f"Saved {len(df)} records to {self.data_file_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
    
    def update_coe_database(self):
        """
        Main method to update COE database with latest data
        """
        print("="*50)
        print("COE Database Update Started")
        print("="*50)
        
        try:
            # Step 1: Fetch latest data from API
            new_data = self.fetch_latest_coe_data()
            
            if new_data is None:
                print("No new data fetched from API")
                return False
            
            # Step 2: Load existing data
            existing_data = self.load_existing_data()
            
            # Step 3: Merge and deduplicate
            updated_data = self.merge_and_deduplicate(existing_data, new_data)
            
            # Step 4: Save updated data
            success = self.save_updated_data(updated_data)
            
            if success:
                print("="*50)
                print("COE Database Update Completed Successfully")
                print(f"Total records in database: {len(updated_data)}")
                print(f"Date range: {updated_data['date'].min()} to {updated_data['date'].max()}")
                print("="*50)
                return True
            else:
                print("Failed to save updated data")
                return False
                
        except Exception as e:
            print(f"Error during database update: {str(e)}")
            return False
    
    def get_latest_exercise_date(self):
        """
        Get the date of the most recent COE exercise in database
        """
        try:
            df = self.load_existing_data()
            if len(df) > 0:
                latest_date = df['date'].max()
                return latest_date
            return None
        except:
            return None
    
    def should_update(self):
        """
        Check if database should be updated based on COE exercise schedule
        """
        latest_date = self.get_latest_exercise_date()
        current_date = datetime.now()
        
        if latest_date is None:
            return True  # No data, should update
        
        # COE exercises are typically on 1st and 15th (or near those dates)
        # Check if it's been more than 16 days since last update
        days_since_update = (current_date - latest_date).days
        
        return days_since_update > 16


def run_coe_update():
    """
    Convenience function to run COE data update
    """
    updater = COEDataUpdater()
    return updater.update_coe_database()


if __name__ == "__main__":
    # Run update when script is executed directly
    run_coe_update()
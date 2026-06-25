import pandas as pd
import os

def clean_dataset():
    # Automatically find the newly downloaded Mendeley dataset
    mendeley_files = [f for f in os.listdir('.') if f.startswith('device2_air_quality') and f.endswith('.csv')]
    
    if not mendeley_files:
        print("ERROR: Cannot find the 'device2_air_quality...' CSV file. Please ensure it is in this folder.")
        return

    RAW_FILE = mendeley_files[0]
    CLEAN_FILE = "cleaned_classrooms.csv"

    print(f"1. Loading real-world IAQ dataset '{RAW_FILE}'...")
    df = pd.read_csv(RAW_FILE, low_memory=False)
    
    # Standardize headers (lowercase and strip spaces)
    df.columns = df.columns.str.lower().str.strip()
    
    # Dynamically find the real sensor columns based on common Mendeley naming conventions
    col_map = {}
    for col in df.columns:
        if 'temp' in col:
            col_map[col] = 'temperature'
        elif 'humid' in col or 'rh' == col:
            col_map[col] = 'humidity'
        elif 'co2' in col:
            col_map[col] = 'co2'

    # Extract only the real sensor data we need
    if not col_map:
        print("\nCRITICAL ERROR: Could not identify the sensor columns!")
        print("Available columns in raw data are:\n", list(df.columns))
        return

    final_df = df[list(col_map.keys())].copy()
    final_df.rename(columns=col_map, inplace=True)
    
    # Drop any rows where the sensor temporarily failed and recorded blanks
    if 'temperature' in final_df.columns and 'co2' in final_df.columns:
        final_df = final_df.dropna(subset=['temperature', 'co2']).copy()
    else:
        print("\nCRITICAL ERROR: Missing Temperature or CO2 data in this file.")
        print("Columns found mapped to:", list(final_df.columns))
        return

    print(f"2. Successfully extracted {len(final_df)} real-world classroom sensor snapshots.")
    print("   Verified Real Data Columns:", list(final_df.columns))
    
    # Save to our clean, lightweight CSV
    final_df.to_csv(CLEAN_FILE, index=False)
    print(f"3. Success! Saved stripped-down, 100% real verified data to '{CLEAN_FILE}'.")

if __name__ == "__main__":
    clean_dataset()

# src/data_fetch.py

import pandas as pd
import os

def load_data():
    # Path to dataset inside the repo
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "spacex_launch_data.csv")
    
    # Load CSV into pandas DataFrame
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    df = load_data()
    print("✅ Dataset loaded successfully!")
    print(df.head())

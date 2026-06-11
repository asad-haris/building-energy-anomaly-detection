import os
import pandas as pd
import numpy as np

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "features", "feature_engineered_data.csv")

# Create output folder if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def feature_engineering():
    # Load cleaned data
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Select ONE meter column
    electricity_col = 'Panther_office_Hannah'

    # Keep only required columns
    df = df[['timestamp', electricity_col]]

    # ================= FEATURE ENGINEERING =================

    # Rolling statistics (7-day = 168 hours)
    df['electricity_rolling_mean'] = df[electricity_col].rolling(168).mean()
    df['electricity_rolling_std'] = df[electricity_col].rolling(168).std()

    # Deviation from baseline
    df['electricity_deviation'] = (
        df[electricity_col] - df['electricity_rolling_mean']
    ) / (df['electricity_rolling_std'] + 1e-5)

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Lag features
    df['electricity_lag1'] = df[electricity_col].shift(1)
    df['electricity_lag24'] = df[electricity_col].shift(24)

    # Drop rows created by rolling/lag NaNs
    engineered_cols = [
        'electricity_rolling_mean',
        'electricity_rolling_std',
        'electricity_deviation',
        'electricity_lag1',
        'electricity_lag24'
    ]
    df = df.dropna(subset=engineered_cols)

    # Save output
    df.to_csv(OUTPUT_PATH, index=False)

    print(" Feature engineering successful!")
    print(" Saved to:", OUTPUT_PATH)
    print(" Rows:", len(df))
    print(" Columns:", df.shape[1])



if __name__ == "__main__":
    feature_engineering()

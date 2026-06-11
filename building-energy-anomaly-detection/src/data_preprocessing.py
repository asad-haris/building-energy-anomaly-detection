import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "electricity_raw.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_data.csv")


def preprocess_data():
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Handle missing values (time-series safe)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Cap outliers at 1st and 99th percentile
    for col in df.columns:
        if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].clip(
                lower=df[col].quantile(0.01),
                upper=df[col].quantile(0.99)
            )

    # Select numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Normalize features to [0,1]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save cleaned data
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Preprocessing complete")
    print(f"✅ Cleaned data saved to: {PROCESSED_DATA_PATH}")


# ---------------- RUN INDIVIDUALLY ----------------
if __name__ == "__main__":
    preprocess_data()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "output", "final_anomalies_ensemble.csv"
)

VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------- 1. TIME SERIES WITH ANOMALIES ----------------
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['Panther_office_Hannah'], label='Electricity', linewidth=1)
plt.scatter(
    df.loc[df['final_anomaly'] == 1, 'timestamp'],
    df.loc[df['final_anomaly'] == 1, 'Panther_office_Hannah'],
    s=25,
    label='Anomaly'
)
plt.title("Electricity Consumption with Detected Anomalies")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "time_series_anomalies.png"))
plt.close()

# ---------------- 2. ANOMALIES BY HOUR ----------------
plt.figure(figsize=(8, 4))
sns.countplot(x='hour', data=df[df['final_anomaly'] == 1])
plt.title("Anomalies by Hour")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "anomalies_by_hour.png"))
plt.close()

# ---------------- 3. WEEKEND VS WEEKDAY ----------------
plt.figure(figsize=(6, 4))
sns.countplot(x='is_weekend', data=df[df['final_anomaly'] == 1])
plt.title("Weekend vs Weekday Anomalies")
plt.xlabel("Is Weekend (1 = Yes)")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "weekend_vs_weekday.png"))
plt.close()

# ---------------- 4. NORMAL VS ANOMALOUS BOX PLOT ----------------
plt.figure(figsize=(6, 4))
sns.boxplot(
    x='final_anomaly',
    y='Panther_office_Hannah',
    data=df
)
plt.title("Normal vs Anomalous Consumption")
plt.xlabel("Anomaly (0 = Normal, 1 = Anomaly)")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "normal_vs_anomalous_boxplot.png"))
plt.close()

# ---------------- 5. MODEL AGREEMENT ----------------
plt.figure(figsize=(6, 4))
sns.countplot(x='anomaly_votes', data=df)
plt.title("Model Agreement on Anomalies")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "model_agreement.png"))
plt.close()

# ---------------- 6. TREND WITH ANOMALIES ----------------
plt.figure(figsize=(15, 5))
plt.plot(
    df['timestamp'],
    df['electricity_rolling_mean'],
    label='Rolling Mean',
    linewidth=2
)
plt.scatter(
    df[df['final_anomaly'] == 1]['timestamp'],
    df[df['final_anomaly'] == 1]['Panther_office_Hannah'],
    s=10,
    label='Anomaly'
)
plt.legend()
plt.title("Anomalies over Electricity Trend")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "trend_with_anomalies.png"))
plt.close()

print("All visualizations generated and saved in:", VIS_DIR)

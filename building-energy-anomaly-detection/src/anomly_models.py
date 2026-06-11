import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "features", "feature_engineered_data.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ISO_PATH = os.path.join(OUTPUT_DIR, "isolation_forest_anomalies.csv")
LOF_PATH = os.path.join(OUTPUT_DIR, "lof_anomalies.csv")
ROBUST_PATH = os.path.join(OUTPUT_DIR, "robust_cov_anomalies.csv")
ENSEMBLE_PATH = os.path.join(OUTPUT_DIR, "final_anomalies_ensemble.csv")

# ---------------- FEATURE COLUMNS ----------------
FEATURE_COLS = [
    'Panther_office_Hannah',
    'electricity_rolling_mean',
    'electricity_rolling_std',
    'electricity_deviation',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'electricity_lag1',
    'electricity_lag24'
]


def run_models_and_ensemble():
    # Load data
    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURE_COLS].fillna(0)

    # =================== MODEL 1: Isolation Forest ===================
    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    df['anomaly_iso'] = iso_forest.fit_predict(X)
    df['is_anomaly'] = (df['anomaly_iso'] == -1).astype(int)

    df.to_csv(ISO_PATH, index=False)
    print("Isolation Forest completed | Anomalies:", df['is_anomaly'].sum())

    # =================== MODEL 2: Local Outlier Factor ===================
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05
    )
    df['anomaly_lof'] = lof.fit_predict(X)
    df['is_anomaly_lof'] = (df['anomaly_lof'] == -1).astype(int)

    df.to_csv(LOF_PATH, index=False)
    print("LOF completed | Anomalies:", df['is_anomaly_lof'].sum())

    # =================== MODEL 3: Robust Covariance ===================
    robust_cov = EllipticEnvelope(
        contamination=0.05,
        random_state=42
    )
    df['anomaly_maha'] = robust_cov.fit_predict(X)
    df['is_anomaly_maha'] = (df['anomaly_maha'] == -1).astype(int)

    df.to_csv(ROBUST_PATH, index=False)
    print("Robust Covariance completed | Anomalies:", df['is_anomaly_maha'].sum())

    # =================== ENSEMBLE VOTING ===================
    df['anomaly_votes'] = (
        df['is_anomaly']
        + df['is_anomaly_lof']
        + df['is_anomaly_maha']
    )

    # Rule: at least 2 models agree
    df['final_anomaly'] = (df['anomaly_votes'] >= 2).astype(int)

    df.to_csv(ENSEMBLE_PATH, index=False)
    print("Ensemble voting completed | Final anomalies:", df['final_anomaly'].sum())
    print("All results saved in:", OUTPUT_DIR)



if __name__ == "__main__":
    run_models_and_ensemble()

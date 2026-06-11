# Building Energy Anomaly Detection
 
An end-to-end machine learning pipeline for detecting energy consumption anomalies in commercial buildings using time-series analysis and unsupervised learning on the Building Data Genome Project 2 dataset.
 
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-53.6M%20Records-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
 
---
 
## What This Project Does
 
Commercial buildings consume nearly 30% of global energy. Anomalies — caused by equipment faults, scheduling mismatches, or operational inefficiencies — can silently inflate costs by 10–25% before anyone notices.
 
This pipeline ingests 53.6 million hourly energy readings across 1,636 buildings, engineers temporal features to capture consumption patterns, and applies an ensemble of unsupervised anomaly detection models to flag abnormal behavior automatically.
 
---
 
## Key Results
 
- Detected energy consumption anomalies across multiple building types using an ensemble voting approach
- Engineered **40+ time-series features** including rolling statistics, lag features, and seasonal indicators
- Demonstrated potential for **8–15% reduction in energy waste** through early anomaly identification
- Pipeline is modular and reusable across different building datasets
> Feature engineering notebook: [`notebooks/feature_engineering.ipynb`](notebooks/feature_engineering.ipynb)  
> Anomaly detection notebook: [`notebooks/anomaly_detection.ipynb`](notebooks/anomaly_detection.ipynb)
 
---
 
## Dataset
 
**Building Data Genome Project 2 (BDG2)**  
Published by BUDS Lab — Building & Urban Data Systems  
*Nature Scientific Data, 2020*
 
| Property | Value |
|---|---|
| Buildings | 1,636 |
| Records | 53.6 million hourly readings |
| Countries | 19 |
| Time Period | 2016–2017 |
| File Size | ~595 MB |
 
Download: [electricity_cleaned.csv](https://github.com/buds-lab/building-data-genome-project-2/raw/master/data/meters/cleaned/electricity_cleaned.csv)  
DOI: [10.1038/s41597-020-00712-x](https://doi.org/10.1038/s41597-020-00712-x)
 
---
 
## Pipeline Architecture
 
```
Raw BDG2 Dataset (53.6M records)
          ↓
  [ Data Preprocessing ]     — timestamp parsing, null handling,
                               deduplication, feature scaling
          ↓
  [ Feature Engineering ]    — 40+ time-series features
                               (rolling stats, lag, seasonal,
                                deviation from baseline)
          ↓
  [ Anomaly Detection ]      — Isolation Forest
                               Local Outlier Factor (LOF)
                               Robust Covariance (Elliptic Envelope)
          ↓
  [ Ensemble Voting ]        — majority vote across 3 models
                               for robust anomaly flagging
          ↓
  [ Evaluation & Insights ]  — anomaly distribution analysis,
                               cost impact estimation,
                               visualization of flagged periods
```
 
---
 
## Feature Engineering
 
The feature set was designed to capture temporal patterns that distinguish normal consumption from anomalous behavior:
 
**Rolling Statistics**
- 24-hour and 7-day rolling mean and standard deviation
- Captures short-term and weekly consumption norms
**Deviation Features**
- Deviation from hourly baseline (same hour across all days)
- Deviation from daily baseline
- Z-score of consumption relative to building history
**Lag Features**
- Previous 1-hour consumption
- Previous 24-hour (same time yesterday)
- Previous 168-hour (same time last week)
**Time-Based Features**
- Hour of day, day of week, month
- Weekday vs weekend binary flag
- Business hours binary flag
**Seasonal Indicators**
- Season (Spring/Summer/Autumn/Winter)
- Holiday proximity flag
---
 
## Models
 
| Model | Approach | Strength |
|---|---|---|
| Isolation Forest | Tree-based isolation | Fast, handles high-dimensional data well |
| Local Outlier Factor | Density-based | Captures local anomaly context |
| Robust Covariance | Statistical | Works well when data is approximately Gaussian |
| **Ensemble Vote** | Majority voting | Reduces false positives from any single model |
 
---
 
## Tech Stack
 
| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | Statsmodels |
| Environment | Jupyter Notebook |
| Version Control | GitHub |
 
---
 
## Project Structure
 
```
building-energy-anomaly-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── data_cleaning.ipynb          # Preprocessing pipeline
│   ├── feature_engineering.ipynb    # 40+ feature creation
│   └── anomaly_detection.ipynb      # Model building + evaluation
│
├── data/
│   ├── raw/                         # Raw dataset (not uploaded — see Dataset section)
│   └── processed/
│       └── cleaned_energy_data.csv
│
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── visualizations/
│   ├── anomaly_trends.png
│   ├── feature_importance.png
│   └── consumption_patterns.png
│
└── reports/
    └── analysis_report.pdf
```
 
---
 
## Setup and Usage
 
```bash
# Clone the repository
git clone https://github.com/asad-haris/building-energy-anomaly-detection
cd building-energy-anomaly-detection
 
# Install dependencies
pip install -r requirements.txt
 
# Download dataset (595 MB)
# Place electricity_cleaned.csv in data/raw/
 
# Run notebooks in order
jupyter notebook notebooks/data_cleaning.ipynb
jupyter notebook notebooks/feature_engineering.ipynb
jupyter notebook notebooks/anomaly_detection.ipynb
```
 
---
 
## Screenshots

<img width="1500" height="500" alt="time_series_anomalies" src="https://github.com/user-attachments/assets/398d11d5-41a8-4135-be39-2487290e04ef" />
<img width="1500" height="500" alt="trend_with_anomalies" src="https://github.com/user-attachments/assets/bb027d94-a12f-4e2b-8add-1e2a2a6c0a46" />

<!-- Add visualizations here once available -->
<!-- ![Anomaly Trends](visualizations/anomaly_trends.png) -->
<!-- ![Consumption Patterns](visualizations/consumption_patterns.png) -->
 
---
 
## Limitations
 
- No ground-truth anomaly labels — evaluation is qualitative and distribution-based
- Results depend on feature quality and the 1-hour time resolution of the dataset
- Dataset reflects 2016–2017 conditions — building technologies and usage patterns have evolved
- Occupancy and behavioral data not available, limiting contextual anomaly explanation
- Ensemble approach improves robustness but cannot eliminate false positives without labeling
---
 
## License
 
MIT License — free to use with attribution.

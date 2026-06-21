"""
preprocessing.py
================
Data loading, cleaning, feature engineering, and window creation
for the UCI Individual Household Electric Power Consumption dataset.

Dataset URL:
  https://archive.ics.uci.edu/ml/machine-learning-databases/00235/
  household_power_consumption.zip

Author : M1 GSIT – Projet BiLSTM Autoencoder
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00235/household_power_consumption.zip"
)
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH      = os.path.join(DATA_DIR, "household_power_consumption.zip")
CSV_PATH      = os.path.join(DATA_DIR, "household_power_consumption.txt")
SCALER_PATH   = os.path.join(DATA_DIR, "scaler.pkl")

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]

WINDOW_SIZE   = 24   # 24-hour sliding window
ANOMALY_RATIO = 0.05 # 5 % of test set injected as synthetic anomalies


# ─────────────────────────────────────────────
# 1. DOWNLOAD
# ─────────────────────────────────────────────
def download_dataset() -> str:
    """Download and unzip the UCI dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print("[INFO] Dataset already present – skipping download.")
        return CSV_PATH

    print("[INFO] Downloading UCI Household Power Consumption dataset …")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("[INFO] Extracting archive …")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    print(f"[INFO] Dataset ready at: {CSV_PATH}")
    return CSV_PATH


# ─────────────────────────────────────────────
# 2. LOAD & CLEAN
# ─────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load raw data, handle missing values ('?'), parse datetime,
    and resample to hourly means.
    """
    print("[INFO] Loading raw data …")
    df = pd.read_csv(
        csv_path,
        sep=";",
        low_memory=False,
        na_values=["?"],   # '?' → NaN
    )

    # ── Datetime parsing ──────────────────────
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True
    )
    df.set_index("datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)

    # ── Numeric conversion ────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[INFO] Raw shape: {df.shape}  |  NaN count: {df.isna().sum().sum()}")

    # ── Interpolation (linear) ────────────────
    df.interpolate(method="linear", inplace=True)
    df.dropna(inplace=True)

    # ── Resample to hourly ────────────────────
    df_hourly = df.resample("H").mean()
    df_hourly.dropna(inplace=True)

    print(f"[INFO] Hourly resampled shape: {df_hourly.shape}")
    return df_hourly


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION & NORMALISATION
# ─────────────────────────────────────────────
def select_and_scale(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    save_scaler: bool = True,
) -> tuple:
    """
    Select relevant features, split train/test, fit MinMaxScaler on train only.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray  (2-D, shape = [N, n_features])
    scaler                         : MinMaxScaler
    train_idx, test_idx            : DatetimeIndex for plotting
    """
    data = df[FEATURES].values
    n    = len(data)
    split = int(n * train_ratio)

    X_train_raw = data[:split]
    X_test_raw  = data[split:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    if save_scaler:
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved → {SCALER_PATH}")

    train_idx = df.index[:split]
    test_idx  = df.index[split:]

    print(f"[INFO] Train samples: {len(X_train_scaled)}  |  Test samples: {len(X_test_scaled)}")
    return X_train_scaled, X_test_scaled, scaler, train_idx, test_idx


# ─────────────────────────────────────────────
# 4. SLIDING WINDOW
# ─────────────────────────────────────────────
def create_sequences(data: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    """
    Transform a 2-D array [N, features] into 3-D windows [N-W, W, features]
    using a sliding window of size `window`.
    """
    seqs = []
    for i in range(len(data) - window):
        seqs.append(data[i : i + window])
    return np.array(seqs, dtype=np.float32)


# ─────────────────────────────────────────────
# 5. SYNTHETIC ANOMALY INJECTION
# ─────────────────────────────────────────────
def inject_anomalies(
    X_test: np.ndarray,
    ratio: float = ANOMALY_RATIO,
    spike_scale: float = 3.5,
    drop_scale:  float = 0.10,
    seed: int = 42,
) -> tuple:
    """
    Inject synthetic anomalies (spikes and drops) into the test set.
    Returns the modified array and a binary label vector (1 = anomaly).

    Parameters
    ----------
    X_test      : 2-D scaled array  [N, features]
    ratio       : fraction of samples to corrupt
    spike_scale : multiplier for spike anomalies (> 1)
    drop_scale  : multiplier for drop anomalies  (< 1)
    seed        : random seed for reproducibility
    """
    rng      = np.random.default_rng(seed)
    X_mod    = X_test.copy()
    labels   = np.zeros(len(X_mod), dtype=int)

    n_anomalies = int(len(X_mod) * ratio)
    anom_idx    = rng.choice(len(X_mod), n_anomalies, replace=False)

    half = n_anomalies // 2
    spike_idx = anom_idx[:half]
    drop_idx  = anom_idx[half:]

    # Spikes: multiply the first feature (Global_active_power) by spike_scale
    X_mod[spike_idx, 0] = np.clip(X_mod[spike_idx, 0] * spike_scale, 0, 1)
    # Drops: reduce all features
    X_mod[drop_idx, :]  = X_mod[drop_idx, :] * drop_scale

    labels[anom_idx] = 1
    print(f"[INFO] Injected {n_anomalies} synthetic anomalies "
          f"({half} spikes + {len(drop_idx)} drops).")
    return X_mod, labels


# ─────────────────────────────────────────────
# FULL PIPELINE (convenience wrapper)
# ─────────────────────────────────────────────
def full_pipeline(inject: bool = True):
    """
    Run the complete preprocessing pipeline and return everything needed
    for model training and evaluation.

    Returns
    -------
    X_train_seq  : np.ndarray  [N_train, W, F]
    X_test_seq   : np.ndarray  [N_test,  W, F]
    y_test       : np.ndarray  [N_test - W]  binary labels (only if inject=True)
    scaler       : fitted MinMaxScaler
    df_hourly    : full hourly DataFrame
    test_idx     : DatetimeIndex for the test portion
    """
    csv  = download_dataset()
    df   = load_and_clean(csv)
    X_tr, X_te, scaler, train_idx, test_idx = select_and_scale(df)

    y_test = None
    if inject:
        X_te, y_test_raw = inject_anomalies(X_te)
        # Align labels with windows (window starts at index 0)
        # We keep the label of the last timestep in each window
        y_test = np.array(
            [y_test_raw[i + WINDOW_SIZE - 1] for i in range(len(X_te) - WINDOW_SIZE)],
            dtype=int,
        )

    X_train_seq = create_sequences(X_tr, WINDOW_SIZE)
    X_test_seq  = create_sequences(X_te, WINDOW_SIZE)

    print(f"[INFO] X_train_seq: {X_train_seq.shape}")
    print(f"[INFO] X_test_seq : {X_test_seq.shape}")

    return X_train_seq, X_test_seq, y_test, scaler, df, test_idx

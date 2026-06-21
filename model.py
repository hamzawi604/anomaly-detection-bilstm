"""
model.py
========
BiLSTM Autoencoder, Simple Autoencoder, and Isolation Forest
for anomaly detection in electrical consumption time series.

Author : M1 GSIT – Projet BiLSTM Autoencoder
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")
BILSTM_PATH      = os.path.join(MODEL_DIR, "bilstm_autoencoder.keras")
SIMPLE_AE_PATH   = os.path.join(MODEL_DIR, "simple_autoencoder.keras")
IFOREST_PATH     = os.path.join(MODEL_DIR, "isolation_forest.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  BiLSTM AUTOENCODER
# ─────────────────────────────────────────────
def build_bilstm_autoencoder(window_size: int, n_features: int) -> Model:
    """
    Architecture:
        Encoder  : Bidirectional LSTM (64 units) → Dense latent (16)
        Decoder  : RepeatVector → LSTM (64 units) → TimeDistributed Dense

    Input shape  : (batch, window_size, n_features)
    Output shape : (batch, window_size, n_features)
    """
    inputs = keras.Input(shape=(window_size, n_features), name="input")

    # ── Encoder ──────────────────────────────
    enc = layers.Bidirectional(
        layers.LSTM(64, activation="tanh", return_sequences=False),
        name="bi_lstm_encoder",
    )(inputs)

    latent = layers.Dense(16, activation="relu", name="latent")(enc)

    # ── Decoder ──────────────────────────────
    dec = layers.RepeatVector(window_size, name="repeat")(latent)

    dec = layers.LSTM(
        64, activation="tanh", return_sequences=True, name="lstm_decoder"
    )(dec)

    outputs = layers.TimeDistributed(
        layers.Dense(n_features), name="output"
    )(dec)

    model = Model(inputs, outputs, name="BiLSTM_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────
# 2.  SIMPLE (DENSE) AUTOENCODER  – baseline
# ─────────────────────────────────────────────
def build_simple_autoencoder(window_size: int, n_features: int) -> Model:
    """
    Flatten → Dense encoder → Dense latent → Dense decoder → Reshape
    Used as a performance baseline vs the BiLSTM model.
    """
    flat_dim = window_size * n_features

    inputs = keras.Input(shape=(window_size, n_features), name="input")
    x      = layers.Flatten()(inputs)

    # Encoder
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64,  activation="relu")(x)
    latent = layers.Dense(16, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(64,  activation="relu")(latent)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(flat_dim, activation="sigmoid")(x)
    outputs = layers.Reshape((window_size, n_features), name="output")(x)

    model = Model(inputs, outputs, name="Simple_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────
# 3.  TRAINING HELPER
# ─────────────────────────────────────────────
def train_model(
    model: Model,
    X_train: np.ndarray,
    epochs: int     = 30,
    batch_size: int = 64,
    val_split: float = 0.10,
    patience: int   = 5,
) -> keras.callbacks.History:
    """
    Train an autoencoder (input == target) with early stopping.

    Parameters
    ----------
    model      : compiled Keras model
    X_train    : shape [N, window, features] — NORMAL samples only
    epochs     : maximum training epochs
    batch_size : mini-batch size
    val_split  : fraction of training data used for validation
    patience   : early stopping patience (epochs without improvement)

    Returns
    -------
    history : Keras History object
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, X_train,          # autoencoder: input == target
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 4.  ISOLATION FOREST  – second baseline
# ─────────────────────────────────────────────
def build_isolation_forest(contamination: float = 0.05) -> IsolationForest:
    """Return a configured Isolation Forest."""
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )


def train_isolation_forest(
    iforest: IsolationForest,
    X_train: np.ndarray,
) -> IsolationForest:
    """
    Flatten windows and fit the Isolation Forest.
    IsolationForest expects 2-D input → flatten the window dimension.
    """
    X_flat = X_train.reshape(len(X_train), -1)
    iforest.fit(X_flat)
    print("[INFO] Isolation Forest trained.")
    return iforest


def predict_isolation_forest(
    iforest: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict anomaly labels (1 = anomaly, 0 = normal).
    IsolationForest returns -1 for anomalies, 1 for inliers.
    We convert to binary: anomaly = 1, normal = 0.
    """
    X_flat = X.reshape(len(X), -1)
    preds  = iforest.predict(X_flat)   # +1 or -1
    return (preds == -1).astype(int)


# ─────────────────────────────────────────────
# 5.  RECONSTRUCTION ERROR & THRESHOLD
# ─────────────────────────────────────────────
def compute_reconstruction_error(
    model: Model,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute per-sample Mean Squared Error between input and reconstruction.

    Returns
    -------
    errors : np.ndarray shape [N,]  — one scalar per window
    """
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    # MSE per sample: mean over (window, features)
    errors = np.mean((X - X_pred) ** 2, axis=(1, 2))
    return errors


def compute_threshold(errors_train: np.ndarray, percentile: float = 95.0) -> float:
    """
    Anomaly threshold = `percentile`-th percentile of training reconstruction errors.
    Samples above this threshold are classified as anomalies.
    """
    threshold = float(np.percentile(errors_train, percentile))
    print(f"[INFO] Threshold ({percentile}th pct) = {threshold:.6f}")
    return threshold


def classify_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary labels: 1 = anomaly, 0 = normal."""
    return (errors > threshold).astype(int)


# ─────────────────────────────────────────────
# 6.  SAVE / LOAD HELPERS
# ─────────────────────────────────────────────
def save_keras_model(model: Model, path: str) -> None:
    model.save(path)
    print(f"[INFO] Model saved → {path}")


def load_keras_model(path: str) -> Model:
    model = keras.models.load_model(path)
    print(f"[INFO] Model loaded ← {path}")
    return model


def save_iforest(iforest: IsolationForest, path: str = IFOREST_PATH) -> None:
    joblib.dump(iforest, path)
    print(f"[INFO] IsolationForest saved → {path}")


def load_iforest(path: str = IFOREST_PATH) -> IsolationForest:
    iforest = joblib.load(path)
    print(f"[INFO] IsolationForest loaded ← {path}")
    return iforest

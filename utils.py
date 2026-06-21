"""
utils.py
========
Evaluation metrics, ROC curves, confusion matrix, and all visualisations
for the BiLSTM Autoencoder anomaly detection project.

Author : M1 GSIT – Projet BiLSTM Autoencoder
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers / Streamlit)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "normal":    "#2196F3",   # blue
    "anomaly":   "#F44336",   # red
    "threshold": "#FF9800",   # orange
    "train":     "#4CAF50",   # green
    "val":       "#9C27B0",   # purple
}


# ─────────────────────────────────────────────
# 1.  METRICS
# ─────────────────────────────────────────────
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors:  np.ndarray | None = None,
    model_name: str = "Model",
) -> dict:
    """
    Print and return a dict of evaluation metrics.

    Parameters
    ----------
    y_true     : ground-truth binary labels [N]
    y_pred     : predicted binary labels    [N]
    errors     : continuous reconstruction errors (for ROC AUC)
    model_name : displayed in the printed header
    """
    report = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"])
    auc    = roc_auc_score(y_true, errors) if errors is not None else None

    metrics = {
        "accuracy":  float(np.mean(y_true == y_pred)),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       auc,
    }

    print(f"\n{'='*55}")
    print(f"  Evaluation – {model_name}")
    print(f"{'='*55}")
    print(report)
    if auc is not None:
        print(f"  ROC AUC Score : {auc:.4f}")
    print(f"{'='*55}\n")

    return metrics


# ─────────────────────────────────────────────
# 2.  TRAINING CURVES
# ─────────────────────────────────────────────
def plot_training_history(history, model_name: str = "BiLSTM_Autoencoder") -> str:
    """Plot train / validation loss curves and save to plots/."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train Loss",      color=PALETTE["train"], lw=2)
    ax.plot(history.history["val_loss"], label="Validation Loss",  color=PALETTE["val"],   lw=2, ls="--")
    ax.set_title(f"Training History – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"training_history_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 3.  RECONSTRUCTION ERROR HISTOGRAM
# ─────────────────────────────────────────────
def plot_error_histogram(
    errors_train: np.ndarray,
    errors_test:  np.ndarray,
    threshold:    float,
    model_name:   str = "BiLSTM_Autoencoder",
) -> str:
    """Plot overlapping histograms of train/test reconstruction errors."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(errors_train, bins=80, alpha=0.55, color=PALETTE["train"],   label="Train errors",  density=True)
    ax.hist(errors_test,  bins=80, alpha=0.55, color=PALETTE["anomaly"], label="Test errors",   density=True)
    ax.axvline(threshold, color=PALETTE["threshold"], lw=2.5, ls="--",
               label=f"Threshold = {threshold:.5f}")

    ax.set_title(f"Reconstruction Error Distribution – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("MSE Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"error_histogram_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 4.  ANOMALY TIMELINE PLOT
# ─────────────────────────────────────────────
def plot_anomaly_timeline(
    df_hourly:   pd.DataFrame,
    test_idx,
    errors_test: np.ndarray,
    threshold:   float,
    y_pred:      np.ndarray,
    window_size: int = 24,
    model_name:  str = "BiLSTM_Autoencoder",
) -> str:
    """
    Plot Global_active_power over time with anomalies highlighted in red.
    """
    # Align window predictions with timestamps
    # Each prediction corresponds to the END of its window
    aligned_idx = test_idx[window_size:]          # skip first `window_size` timestamps
    n = min(len(aligned_idx), len(y_pred))
    aligned_idx = aligned_idx[:n]
    y_pred_plot = y_pred[:n]
    errors_plot = errors_test[:n]

    power_series = df_hourly["Global_active_power"].loc[aligned_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # ── Top: Global Active Power ─────────────
    ax1.plot(aligned_idx, power_series.values,
             color=PALETTE["normal"], lw=0.8, label="Global Active Power (kW)")
    anom_mask = y_pred_plot == 1
    ax1.scatter(aligned_idx[anom_mask], power_series.values[anom_mask],
                color=PALETTE["anomaly"], s=20, zorder=5, label="Anomaly")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title(f"Consumption & Detected Anomalies – {model_name}",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.25)

    # ── Bottom: Reconstruction error ─────────
    ax2.plot(aligned_idx, errors_plot, color="#607D8B", lw=0.7, label="Reconstruction Error")
    ax2.axhline(threshold, color=PALETTE["threshold"], lw=2, ls="--",
                label=f"Threshold ({threshold:.5f})")
    ax2.fill_between(aligned_idx, 0, errors_plot,
                     where=(errors_plot > threshold),
                     color=PALETTE["anomaly"], alpha=0.30, label="Above Threshold")
    ax2.set_ylabel("MSE Error")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.25)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"anomaly_timeline_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 5.  ROC CURVE
# ─────────────────────────────────────────────
def plot_roc_curves(results: dict) -> str:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    results : { model_name: {"y_true": …, "errors": …} }
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for (name, data), color in zip(results.items(), colors):
        y_true  = data["y_true"]
        scores  = data["errors"]          # higher score → more anomalous
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc         = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 6.  CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "BiLSTM_Autoencoder",
) -> str:
    """Plot and save a seaborn heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 7.  METRICS COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_metrics_comparison(all_metrics: dict) -> str:
    """
    Bar chart comparing F1, Precision, Recall, AUC across models.

    Parameters
    ----------
    all_metrics : { model_name: { "f1": …, "precision": …, "recall": …, "auc": … } }
    """
    metric_keys = ["f1", "precision", "recall", "auc"]
    model_names = list(all_metrics.keys())
    x           = np.arange(len(metric_keys))
    width        = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2196F3", "#F44336", "#4CAF50"]

    for i, (name, metrics) in enumerate(all_metrics.items()):
        vals = [metrics.get(k, 0) or 0 for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metric_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison – All Models", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 8.  FEATURE CORRELATION HEATMAP  (EDA)
# ─────────────────────────────────────────────
def plot_feature_correlation(df: pd.DataFrame, features: list) -> str:
    """Plot feature-feature Pearson correlation heatmap."""
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "feature_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 9.  CONSUMPTION OVERVIEW  (EDA)
# ─────────────────────────────────────────────
def plot_consumption_overview(df: pd.DataFrame) -> str:
    """Line plots of all four features over the full timeline."""
    features = ["Global_active_power", "Global_reactive_power",
                "Voltage", "Global_intensity"]
    colors   = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, feat, color in zip(axes, features, colors):
        ax.plot(df.index, df[feat], lw=0.6, color=color)
        ax.set_ylabel(feat.replace("_", " "), fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_title("Household Electrical Consumption Overview", fontsize=13, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "consumption_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path

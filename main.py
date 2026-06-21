"""
main.py
=======
End-to-end pipeline:
  1. Preprocessing  (download, clean, normalise, window)
  2. Training       (BiLSTM AE, Simple AE, Isolation Forest)
  3. Evaluation     (ROC, AUC, Confusion Matrix)
  4. Visualisation  (all plots saved to plots/)

Run:
    python main.py

Author : M1 GSIT – Projet BiLSTM Autoencoder
"""

import os
import numpy as np
import pandas as pd

# ── Local modules ─────────────────────────────
from preprocessing import (
    full_pipeline,
    FEATURES,
    WINDOW_SIZE,
)
from model import (
    build_bilstm_autoencoder,
    build_simple_autoencoder,
    build_isolation_forest,
    train_model,
    train_isolation_forest,
    predict_isolation_forest,
    compute_reconstruction_error,
    compute_threshold,
    classify_anomalies,
    save_keras_model,
    save_iforest,
    BILSTM_PATH,
    SIMPLE_AE_PATH,
    IFOREST_PATH,
)
from utils import (
    evaluate_predictions,
    plot_training_history,
    plot_error_histogram,
    plot_anomaly_timeline,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_feature_correlation,
    plot_consumption_overview,
    PLOTS_DIR,
)


# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════
EPOCHS      = 50      
BATCH_SIZE  = 64
PATIENCE    = 7
THRESHOLD_P = 95     


# ═══════════════════════════════════════════════
# STEP 1 – PREPROCESSING
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 – DATA PREPROCESSING")
print("═"*60)

X_train, X_test, y_test, scaler, df_hourly, test_idx = full_pipeline(inject=True)

n_features = X_train.shape[2]

print(f"\n  Window size    : {WINDOW_SIZE} hours")
print(f"  Features used  : {FEATURES}")
print(f"  Train windows  : {len(X_train)}")
print(f"  Test  windows  : {len(X_test)}")
print(f"  Anomaly labels : {y_test.sum()} / {len(y_test)}")


# ── EDA Plots ────────────────────────────────────
print("\n[EDA] Generating overview plots …")
plot_consumption_overview(df_hourly)
plot_feature_correlation(df_hourly, FEATURES)


# ═══════════════════════════════════════════════
# STEP 2 – BiLSTM AUTOENCODER
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 – BiLSTM AUTOENCODER TRAINING")
print("═"*60)

bilstm = build_bilstm_autoencoder(WINDOW_SIZE, n_features)
bilstm.summary()

history_bilstm = train_model(
    bilstm, X_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
)

save_keras_model(bilstm, BILSTM_PATH)
plot_training_history(history_bilstm, model_name="BiLSTM_AE")


# ── Compute errors & threshold ────────────────
print("\n[INFO] Computing reconstruction errors …")
errors_train_bilstm = compute_reconstruction_error(bilstm, X_train)
errors_test_bilstm  = compute_reconstruction_error(bilstm, X_test)

threshold_bilstm = compute_threshold(errors_train_bilstm, THRESHOLD_P)
y_pred_bilstm    = classify_anomalies(errors_test_bilstm, threshold_bilstm)

plot_error_histogram(errors_train_bilstm, errors_test_bilstm,
                     threshold_bilstm, model_name="BiLSTM_AE")


# ═══════════════════════════════════════════════
# STEP 3 – SIMPLE AUTOENCODER  (baseline)
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 – SIMPLE AUTOENCODER TRAINING  (baseline)")
print("═"*60)

simple_ae = build_simple_autoencoder(WINDOW_SIZE, n_features)
simple_ae.summary()

history_simple = train_model(
    simple_ae, X_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
)

save_keras_model(simple_ae, SIMPLE_AE_PATH)
plot_training_history(history_simple, model_name="Simple_AE")

errors_train_simple = compute_reconstruction_error(simple_ae, X_train)
errors_test_simple  = compute_reconstruction_error(simple_ae, X_test)
threshold_simple    = compute_threshold(errors_train_simple, THRESHOLD_P)
y_pred_simple       = classify_anomalies(errors_test_simple, threshold_simple)

plot_error_histogram(errors_train_simple, errors_test_simple,
                     threshold_simple, model_name="Simple_AE")


# ═══════════════════════════════════════════════
# STEP 4 – ISOLATION FOREST  (baseline)
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 – ISOLATION FOREST  (baseline)")
print("═"*60)

iforest = build_isolation_forest(contamination=0.05)
iforest = train_isolation_forest(iforest, X_train)
save_iforest(iforest, IFOREST_PATH)

y_pred_iforest = predict_isolation_forest(iforest, X_test)


X_test_flat  = X_test.reshape(len(X_test), -1)
if_scores    = -iforest.decision_function(X_test_flat)   # negate


# ═══════════════════════════════════════════════
# STEP 5 – EVALUATION & PLOTS
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 5 – EVALUATION")
print("═"*60)

metrics_bilstm = evaluate_predictions(
    y_test, y_pred_bilstm, errors_test_bilstm, model_name="BiLSTM Autoencoder")

metrics_simple = evaluate_predictions(
    y_test, y_pred_simple, errors_test_simple, model_name="Simple Autoencoder")

metrics_iforest = evaluate_predictions(
    y_test, y_pred_iforest, if_scores, model_name="Isolation Forest")


# ── Confusion matrices ────────────────────────
plot_confusion_matrix(y_test, y_pred_bilstm,  model_name="BiLSTM_AE")
plot_confusion_matrix(y_test, y_pred_simple,  model_name="Simple_AE")
plot_confusion_matrix(y_test, y_pred_iforest, model_name="IsolationForest")


# ── ROC Curves ────────────────────────────────
roc_data = {
    "BiLSTM AE":       {"y_true": y_test, "errors": errors_test_bilstm},
    "Simple AE":       {"y_true": y_test, "errors": errors_test_simple},
    "Isolation Forest":{"y_true": y_test, "errors": if_scores},
}
plot_roc_curves(roc_data)


# ── Metrics bar chart ─────────────────────────
all_metrics = {
    "BiLSTM AE":        metrics_bilstm,
    "Simple AE":        metrics_simple,
    "Isolation Forest": metrics_iforest,
}
plot_metrics_comparison(all_metrics)


# ── Anomaly timeline ──────────────────────────
plot_anomaly_timeline(
    df_hourly, test_idx,
    errors_test_bilstm, threshold_bilstm,
    y_pred_bilstm,
    window_size=WINDOW_SIZE,
    model_name="BiLSTM_AE",
)


# ═══════════════════════════════════════════════
# STEP 6 – SUMMARY TABLE
# ═══════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 – FINAL SUMMARY")
print("═"*60)

summary = pd.DataFrame({
    "Model":     ["BiLSTM AE", "Simple AE", "Isolation Forest"],
    "Precision": [
        round(metrics_bilstm["precision"], 4),
        round(metrics_simple["precision"], 4),
        round(metrics_iforest["precision"], 4),
    ],
    "Recall": [
        round(metrics_bilstm["recall"], 4),
        round(metrics_simple["recall"], 4),
        round(metrics_iforest["recall"], 4),
    ],
    "F1 Score": [
        round(metrics_bilstm["f1"], 4),
        round(metrics_simple["f1"], 4),
        round(metrics_iforest["f1"], 4),
    ],
    "AUC": [
        round(metrics_bilstm["auc"] or 0, 4),
        round(metrics_simple["auc"] or 0, 4),
        round(metrics_iforest["auc"] or 0, 4),
    ],
})

print(summary.to_string(index=False))
summary.to_csv(os.path.join("outputs", "metrics_summary.csv"), index=False)

print(f"\n[DONE] All plots saved in   → {PLOTS_DIR}/")
print("[DONE] Metrics CSV saved    → outputs/metrics_summary.csv")
print("[DONE] Trained models saved → models/")
"""
app.py  –  Streamlit Dashboard
================================
Anomaly Detection in Electrical Consumption – BiLSTM Autoencoder

Run:
    streamlit run app.py

Author : M1 GSIT – Projet BiLSTM Autoencoder
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import joblib
import tensorflow as tf
from tensorflow import keras

# ── Local modules ─────────────────────────────
from preprocessing import (
    load_and_clean,
    select_and_scale,
    create_sequences,
    inject_anomalies,
    FEATURES,
    WINDOW_SIZE,
    SCALER_PATH,
    CSV_PATH,
    DATA_DIR,
)
from model import (
    compute_reconstruction_error,
    compute_threshold,
    classify_anomalies,
    BILSTM_PATH,
    SIMPLE_AE_PATH,
    IFOREST_PATH,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Anomaly Detection – Electrical Consumption",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #2196F3;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: white;
    }
    .metric-label { font-size: 12px; color: #90CAF9; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #FFFFFF; }
    .anomaly-badge {
        background: #F44336;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: bold;
    }
    .normal-badge {
        background: #4CAF50;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS – Cached loaders
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bilstm():
    if os.path.exists(BILSTM_PATH):
        return keras.models.load_model(BILSTM_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_simple_ae():
    if os.path.exists(SIMPLE_AE_PATH):
        return keras.models.load_model(SIMPLE_AE_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_iforest_model():
    if os.path.exists(IFOREST_PATH):
        return joblib.load(IFOREST_PATH)
    return None

@st.cache_resource(show_spinner=False)
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def run_detection(df: pd.DataFrame, model_choice: str, threshold_pct: int, inject: bool):
    """Core detection pipeline. Returns (errors, threshold, labels, timestamps)."""

    scaler = load_scaler()
    if scaler is None:
        st.error("⚠️ Scaler not found. Please run `main.py` first to train the models.")
        return None, None, None, None

    # Scale with the pre-fitted scaler
    data   = df[FEATURES].values
    scaled = scaler.transform(data)

    if inject:
        scaled, _ = inject_anomalies(scaled, ratio=0.05)

    sequences  = create_sequences(scaled, WINDOW_SIZE)
    timestamps = df.index[WINDOW_SIZE:]

    if model_choice == "BiLSTM Autoencoder":
        model = load_bilstm()
        if model is None:
            st.error("⚠️ BiLSTM model not found. Run `main.py` first.")
            return None, None, None, None
        errors = compute_reconstruction_error(model, sequences)

    elif model_choice == "Simple Autoencoder":
        model = load_simple_ae()
        if model is None:
            st.error("⚠️ Simple AE not found. Run `main.py` first.")
            return None, None, None, None
        errors = compute_reconstruction_error(model, sequences)

    else:  # Isolation Forest
        iforest = load_iforest_model()
        if iforest is None:
            st.error("⚠️ Isolation Forest not found. Run `main.py` first.")
            return None, None, None, None
        X_flat = sequences.reshape(len(sequences), -1)
        errors = -iforest.decision_function(X_flat)

    threshold = float(np.percentile(errors, threshold_pct))
    labels    = classify_anomalies(errors, threshold)

    return errors, threshold, labels, timestamps[:len(errors)]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/150px-Tux.svg.png",
             width=60)
    st.title("⚡ Anomaly Detection")
    st.caption("M1 GSIT — BiLSTM Autoencoder Project")
    st.divider()

    st.subheader("📂 Data Source")
    data_source = st.radio(
        "Choose input",
        ["Use Pre-loaded Dataset", "Upload CSV / TXT File"],
        index=0,
    )

    uploaded_file = None
    if data_source == "Upload CSV / TXT File":
        uploaded_file = st.file_uploader(
            "Upload household_power_consumption.txt",
            type=["txt", "csv"],
        )

    st.divider()

    st.subheader("🤖 Model")
    model_choice = st.selectbox(
        "Detection Model",
        ["BiLSTM Autoencoder", "Simple Autoencoder", "Isolation Forest"],
        index=0,
    )

    st.divider()

    st.subheader("⚙️ Parameters")
    threshold_pct = st.slider(
        "Anomaly Threshold (percentile)",
        min_value=80, max_value=99, value=95, step=1,
    )
    inject_anomalies_flag = st.checkbox("Inject synthetic anomalies", value=True)

    st.divider()
    st.subheader("📅 Date Range")
    use_date_filter = st.checkbox("Filter date range", value=False)

    run_button = st.button("🚀 Run Detection", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("⚡ Electrical Consumption Anomaly Detection")
st.markdown(
    "**BiLSTM Autoencoder** trained exclusively on normal patterns. "
    "Anomalies are detected by elevated reconstruction error."
)

tab_detect, tab_eda, tab_about = st.tabs(["🔍 Detection", "📊 EDA & Overview", "ℹ️ About"])


# ══════════════════════════════════════
# TAB 1 – DETECTION
# ══════════════════════════════════════
with tab_detect:
    if not run_button:
        st.info("👈 Configure the settings in the sidebar and click **Run Detection**.")

    else:
        # ── Load data ──────────────────────────────
        with st.spinner("Loading and preprocessing data …"):
            try:
                if data_source == "Upload CSV / TXT File" and uploaded_file is not None:
                    tmp_path = os.path.join(DATA_DIR, "uploaded.txt")
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    df = load_and_clean(tmp_path)
                else:
                    if not os.path.exists(CSV_PATH):
                        st.error(
                            "Dataset not found. Run `python main.py` first to download it, "
                            "or upload a file manually."
                        )
                        st.stop()
                    df = load_and_clean(CSV_PATH)

                if use_date_filter:
                    min_date = df.index.min().date()
                    max_date = df.index.max().date()
                    col1, col2 = st.columns(2)
                    start_date = col1.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                    end_date   = col2.date_input("End date",   max_date, min_value=min_date, max_value=max_date)
                    df = df.loc[str(start_date):str(end_date)]

            except Exception as e:
                st.error(f"Data loading error: {e}")
                st.stop()

        # ── Run model ──────────────────────────────
        with st.spinner(f"Running {model_choice} …"):
            errors, threshold, labels, timestamps = run_detection(
                df, model_choice, threshold_pct, inject_anomalies_flag
            )

        if errors is None:
            st.stop()

        # ── KPI Cards ──────────────────────────────
        n_anomalies = int(labels.sum())
        anom_pct    = n_anomalies / len(labels) * 100
        mean_err    = float(errors.mean())
        max_err     = float(errors.max())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Anomalies Detected", f"{n_anomalies:,}")
        c2.metric("📊 Anomaly Rate",        f"{anom_pct:.2f}%")
        c3.metric("📉 Mean Recon. Error",   f"{mean_err:.5f}")
        c4.metric("📈 Threshold",           f"{threshold:.5f}")

        st.divider()

        # ── Consumption + anomaly plot ──────────────
        power = df["Global_active_power"].values[WINDOW_SIZE:WINDOW_SIZE + len(labels)]
        ts    = timestamps[:len(labels)]

        anom_mask   = labels == 1
        normal_mask = labels == 0

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Global Active Power (kW) – Anomalies highlighted in red",
                            "Reconstruction Error"],
            vertical_spacing=0.10,
            row_heights=[0.6, 0.4],
        )

        # Power line
        fig.add_trace(go.Scatter(
            x=ts[normal_mask], y=power[normal_mask],
            mode="lines", name="Normal",
            line=dict(color="#2196F3", width=1),
        ), row=1, col=1)

        # Anomaly markers
        fig.add_trace(go.Scatter(
            x=ts[anom_mask], y=power[anom_mask],
            mode="markers", name="Anomaly",
            marker=dict(color="#F44336", size=6, symbol="circle"),
        ), row=1, col=1)

        # Reconstruction error
        fig.add_trace(go.Scatter(
            x=ts, y=errors[:len(ts)],
            mode="lines", name="Recon. Error",
            line=dict(color="#607D8B", width=0.8),
            fill="tozeroy", fillcolor="rgba(96,125,139,0.15)",
        ), row=2, col=1)

        # Threshold line
        fig.add_hline(
            y=threshold, row=2, col=1,
            line_dash="dash", line_color="#FF9800",
            annotation_text=f"Threshold={threshold:.4f}",
            annotation_position="top right",
        )

        fig.update_layout(
            height=550, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Error histogram ────────────────────────
        st.subheader("📊 Reconstruction Error Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=errors, nbinsx=100, name="Reconstruction Errors",
            marker_color="#2196F3", opacity=0.7,
        ))
        fig2.add_vline(
            x=threshold, line_dash="dash", line_color="#FF9800",
            annotation_text=f"Threshold = {threshold:.5f}",
            annotation_position="top right",
        )
        fig2.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="MSE Error", yaxis_title="Count",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Anomaly table ──────────────────────────
        st.subheader("🗂️ Detected Anomaly Events")
        df_result = pd.DataFrame({
            "Timestamp":   ts,
            "Power (kW)":  power[:len(ts)],
            "Recon. Error": errors[:len(ts)],
            "Status": ["🔴 Anomaly" if l == 1 else "🟢 Normal" for l in labels[:len(ts)]],
        })
        anomaly_rows = df_result[df_result["Status"] == "🔴 Anomaly"].copy()

        if len(anomaly_rows):
            st.dataframe(
                anomaly_rows.sort_values("Recon. Error", ascending=False)
                    .reset_index(drop=True)
                    .head(100),
                use_container_width=True,
            )
            csv_bytes = anomaly_rows.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Anomaly Report (CSV)",
                data=csv_bytes,
                file_name="anomaly_report.csv",
                mime="text/csv",
            )
        else:
            st.success("No anomalies detected in the selected range.")


# ══════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis")

    @st.cache_data(show_spinner=False)
    def get_eda_data():
        if not os.path.exists(CSV_PATH):
            return None
        return load_and_clean(CSV_PATH)

    df_eda = get_eda_data()

    if df_eda is None:
        st.info("Run `main.py` first to download the dataset.")
    else:
        st.write(f"**Dataset:** {len(df_eda):,} hourly records | "
                 f"{df_eda.index.min().date()} → {df_eda.index.max().date()}")

        col_a, col_b = st.columns(2)

        with col_a:
            feat = st.selectbox("Select feature", FEATURES, index=0)

        with col_b:
            resample_opt = st.selectbox("Resample", ["Hourly (H)", "Daily (D)", "Weekly (W)"], index=1)

        rs_map = {"Hourly (H)": "H", "Daily (D)": "D", "Weekly (W)": "W"}
        rs_key = rs_map[resample_opt]

        s = df_eda[feat].resample(rs_key).mean()

        fig_eda = px.line(s, x=s.index, y=s.values,
                          labels={"x": "Date", "y": feat},
                          title=f"{feat} – {resample_opt} Average",
                          template="plotly_dark")
        fig_eda.update_traces(line_width=1.2)
        st.plotly_chart(fig_eda, use_container_width=True)

        st.subheader("📋 Statistical Summary")
        st.dataframe(df_eda[FEATURES].describe().round(4), use_container_width=True)

        st.subheader("🔗 Feature Correlation")
        corr = df_eda[FEATURES].corr().round(3)
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            aspect="auto", template="plotly_dark",
            title="Pearson Correlation Matrix",
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════
# TAB 3 – ABOUT
# ══════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About this Project")
    st.markdown("""
    ### 🎓 Projet M1 GSIT – Mme Amel KHEITER · 2025/2026

    **Title:** Détection d'Anomalies de Consommation Électrique par BiLSTM-Autoencoder

    ---

    ### 🏗️ Architecture: BiLSTM Autoencoder

    ```
    Input  (batch, 24, 4)
       │
       ▼
    Bidirectional LSTM (64 units)   ← Encoder
       │
       ▼
    Dense (16)                      ← Latent Space
       │
       ▼
    RepeatVector (24)
       │
       ▼
    LSTM (64 units)                 ← Decoder
       │
       ▼
    TimeDistributed Dense (4)
       │
       ▼
    Output (batch, 24, 4)
    ```

    ---

    ### 📊 Dataset
    - **Source:** UCI Machine Learning Repository
    - **Name:** Individual Household Electric Power Consumption
    - **Size:** ~2M minute-level measurements (2006–2010)
    - **Resampled:** Hourly averages (→ ~35 000 records)
    - **Features used:** Global Active Power, Global Reactive Power, Voltage, Global Intensity

    ---

    ### 🔍 Anomaly Detection Pipeline
    1. Train BiLSTM AE only on **normal** patterns
    2. Compute **reconstruction error (MSE)** on test data
    3. Threshold = **95th percentile** of training errors
    4. Samples exceeding threshold → **anomaly**

    ---

    ### 📁 Project Structure
    ```
    anomaly_detection/
    ├── app.py              ← Streamlit dashboard (this file)
    ├── main.py             ← Full training & evaluation pipeline
    ├── model.py            ← BiLSTM AE, Simple AE, Isolation Forest
    ├── preprocessing.py    ← Data loading, cleaning, windowing
    ├── utils.py            ← Plots, metrics, evaluation
    ├── requirements.txt    ← Python dependencies
    ├── data/               ← Raw dataset (auto-downloaded)
    ├── models/             ← Saved .keras models
    ├── plots/              ← Saved PNG figures
    └── outputs/            ← CSV reports
    ```
    """)

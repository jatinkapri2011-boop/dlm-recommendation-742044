import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="UPI Fraud Risk ANN Dashboard",
    layout="wide"
)

st.title("🚨 UPI Fraud Risk Prediction Dashboard (2018–2024)")
st.markdown("### Deep Learning for Managers Project")

# -------------------------------------------------
# LOAD DATA AUTOMATICALLY FROM REPO
# -------------------------------------------------
@st.cache_data
def load_data():

    csv_files = [f for f in os.listdir() if f.endswith(".csv")]
    if not csv_files:
        st.error("No CSV files found in repository.")
        st.stop()

    all_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        volume_col = [c for c in df.columns if "Volume" in c][0]
        value_col = [c for c in df.columns if "Value" in c][0]
        month_col = [c for c in df.columns if "Month" in c][0]

        df = df.rename(columns={
            volume_col: "Volume_Million",
            value_col: "Value_Crore",
            month_col: "Month"
        })

        df["Volume_Million"] = (
            df["Volume_Million"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        df["Value_Crore"] = (
            df["Value_Crore"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        df["Volume_Million"] = pd.to_numeric(df["Volume_Million"], errors="coerce")
        df["Value_Crore"] = pd.to_numeric(df["Value_Crore"], errors="coerce")

        df["Date"] = pd.to_datetime(df["Month"], format="%B-%Y", errors="coerce")

        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna()
    df = df.sort_values("Date")

    df["Volume_Growth"] = df["Volume_Million"].pct_change()
    df["Value_Growth"] = df["Value_Crore"].pct_change()
    df = df.dropna()

    threshold = df["Volume_Growth"].quantile(0.75)
    df["Fraud_Risk"] = (df["Volume_Growth"] > threshold).astype(int)

    df["Year"] = df["Date"].dt.year

    return df


df = load_data()

st.success(f"Dataset Covers: {df['Date'].min().date()} to {df['Date'].max().date()}")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Evaluation"])

# =================================================
# TAB 1 – DATA OVERVIEW
# =================================================
with tab1:

    st.subheader("📅 Date Range Filter")

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    date_range = st.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    df_filtered = df[
        (df["Date"] >= date_range[0]) &
        (df["Date"] <= date_range[1])
    ]

    # YEAR SUMMARY
    st.subheader("📊 Year-wise Summary")

    yearly_summary = (
        df_filtered
        .groupby("Year")
        .agg({
            "Volume_Million": "sum",
            "Value_Crore": "sum",
            "Fraud_Risk": "sum"
        })
        .reset_index()
    )

    yearly_summary = yearly_summary.rename(columns={
        "Volume_Million": "Total Volume (Million)",
        "Value_Crore": "Total Value (Crore)",
        "Fraud_Risk": "High-Risk Months"
    })

    st.dataframe(yearly_summary)

    # TIME SERIES
    st.subheader("📈 UPI Volume Trend")

    fig, ax = plt.subplots()
    ax.plot(df_filtered["Date"], df_filtered["Volume_Million"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (Million)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # CLASS DISTRIBUTION
    st.subheader("⚠ Fraud Risk Distribution")
    st.bar_chart(df_filtered["Fraud_Risk"].value_counts())

    # FULL DATA
    st.subheader("📂 Full Dataset")
    st.dataframe(df_filtered)


# =================================================
# TAB 2 – MODEL TRAINING
# =================================================
with tab2:

    st.sidebar.header("⚙ Hyperparameters")

    neurons = st.sidebar.slider("Neurons", 8, 128, 32)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    use_class_weight = st.sidebar.checkbox("Handle Class Imbalance", value=True)

    X = df_filtered[["Volume_Growth", "Value_Growth"]]
    y = df_filtered["Fraud_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(neurons, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    if st.button("🚀 Train Model"):

        class_weight = None
        if use_class_weight:
            weight_0 = len(y) / (2 * sum(y == 0))
            weight_1 = len(y) / (2 * sum(y == 1))
            class_weight = {0: weight_0, 1: weight_1}

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
            class_weight=class_weight
        )

        st.success("Model Training Complete")

        # Loss Curve
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["loss"], label="Train Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.legend()
        st.pyplot(fig2)

        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["y_prob"] = y_pred_prob


# =================================================
# TAB 3 – EVALUATION
# =================================================
with tab3:

    if "y_pred" in st.session_state:

        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        y_prob = st.session_state["y_prob"]

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
        st.pyplot(fig3)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        roc = roc_auc_score(y_test, y_pred)
        st.write(f"ROC-AUC Score: {roc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig4, ax4 = plt.subplots()
        ax4.plot(fpr, tpr)
        ax4.plot([0,1],[0,1],'--')
        ax4.set_title("ROC Curve")
        st.pyplot(fig4)

    else:
        st.info("Train the model first in the Model Training tab.")

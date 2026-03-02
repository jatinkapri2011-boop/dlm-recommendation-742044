import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="UPI Fraud Risk ANN", layout="wide")

st.title("🚨 UPI Fraud Risk ANN Hyperparameter Tuning Dashboard")

# ------------------------------
# Load All CSV Files Automatically
# ------------------------------
csv_files = [f for f in os.listdir() if f.endswith(".csv")]

if not csv_files:
    st.error("No CSV files found in repository.")
    st.stop()

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    volume_col = [col for col in df.columns if "Volume" in col][0]
    value_col = [col for col in df.columns if "Value" in col][0]

    df = df.rename(columns={
        volume_col: "Volume_Million",
        value_col: "Value_Crore"
    })

    # Remove commas
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

    all_dfs.append(df)

# Merge all years
df = pd.concat(all_dfs, ignore_index=True)
df = df.dropna()

# ------------------------------
# Feature Engineering
# ------------------------------
df["Volume_Growth"] = df["Volume_Million"].pct_change()
df["Value_Growth"] = df["Value_Crore"].pct_change()
df = df.dropna()

# Fraud Risk Label (Top 25% growth spikes)
threshold = df["Volume_Growth"].quantile(0.75)
df["Fraud_Risk"] = (df["Volume_Growth"] > threshold).astype(int)

st.success("✅ All yearly UPI files loaded and processed automatically.")

# ------------------------------
# Sidebar Hyperparameter Controls
# ------------------------------
st.sidebar.header("⚙ ANN Hyperparameters")

neurons = st.sidebar.slider("Neurons", 8, 128, 32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)

# ------------------------------
# Prepare Data
# ------------------------------
X = df[["Volume_Growth", "Value_Growth"]]
y = df["Fraud_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Build ANN Model
# ------------------------------
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

# ------------------------------
# Train Model
# ------------------------------
if st.button("🚀 Train Model"):

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=0
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Metrics
    st.subheader("📊 Model Performance")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    roc = roc_auc_score(y_test, y_pred)
    st.write(f"ROC-AUC Score: {roc:.4f}")

    st.success("Model training complete!")

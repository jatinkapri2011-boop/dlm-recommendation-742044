import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

st.title("UPI Fraud Risk ANN Hyperparameter Tuning Dashboard")

# ---- Upload Data ----
uploaded_file = st.file_uploader(
    "Upload UPI Data File",
    type=["csv", "xlsx"],
    key="upi_file"
)

if uploaded_file is not None:

    file_name = uploaded_file.name

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file format")
            st.stop()

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()



    # cleaning + feature engineering here




if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Auto-detect correct column names
    volume_col = [col for col in df.columns if "Volume" in col][0]
    value_col = [col for col in df.columns if "Value" in col][0]

    # Rename for consistency
    df = df.rename(columns={
        volume_col: "Volume_Million",
        value_col: "Value_Crore"
    })

    # Feature Engineering
    df['Volume_Growth'] = df['Volume_Million'].pct_change()
    df['Value_Growth'] = df['Value_Crore'].pct_change()
    df = df.dropna()


    df = df.dropna()

    threshold = df['Volume_Growth'].quantile(0.75)
    df['Fraud_Risk'] = (df['Volume_Growth'] > threshold).astype(int)

    features = ['Volume_Million','Value_Crore','Volume_Growth','Value_Growth']
    X = df[features]
    y = df['Fraud_Risk']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ---- Sidebar Controls ----
    st.sidebar.header("Hyperparameters")

    neurons1 = st.sidebar.slider("Layer 1 Neurons", 8, 128, 32)
    neurons2 = st.sidebar.slider("Layer 2 Neurons", 4, 64, 16)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
    epochs = st.sidebar.slider("Epochs", 10, 200, 100)
    batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32])

    # ---- Handle Class Imbalance ----
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(enumerate(class_weights))

    if st.button("Train Model"):

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(neurons1, activation='relu'),
            Dropout(dropout_rate),
            Dense(neurons2, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weight_dict,
            verbose=0
        )

        # Predictions
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)

        # ---- Results ----
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax2.plot([0,1],[0,1],'--')
        ax2.legend()
        st.pyplot(fig2)

        st.success(f"AUC Score: {auc_score:.4f}")

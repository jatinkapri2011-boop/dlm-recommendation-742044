# 💳 UPI Fraud Risk Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project develops an Artificial Neural Network (ANN) model to predict high-risk transaction growth periods in UPI (Unified Payments Interface) using monthly NPCI data (2018–2024).

As UPI transactions crossed 100+ billion transactions annually, rapid growth phases increase potential fraud exposure. This model helps fintech firms proactively identify high-risk months.

---

## 🎯 Problem Statement

With exponential growth in UPI transaction volume and value, fraud risk exposure increases during rapid expansion phases.

Goal:
Develop an ANN-based fraud risk classification system that:

- Identifies high-risk transaction growth months
- Handles class imbalance
- Provides hyperparameter tuning via Streamlit dashboard
- Generates managerial insights for fintech risk management

---

## 📊 Dataset

Source: NPCI Official UPI Product Statistics  
Link: https://www.npci.org.in/what-we-do/upi/product-statistics

Data Used:
- Monthly UPI Volume (in Million)
- Monthly UPI Value (in Crore ₹)
- Financial Years: 2018–2024

Data preprocessing included:
- Financial year to calendar year conversion
- Chronological sorting
- Growth rate feature engineering

---

## ⚙️ Feature Engineering

Created fraud risk indicators using:

- Volume Growth Rate
- Value Growth Rate
- Transaction Volume
- Transaction Value

High-Risk Label Definition:
Top 25% growth months classified as High Fraud Risk (1)
Remaining months classified as Normal (0)

---

## 🧠 ANN Model Architecture

- Input Layer
- Dense Layer (ReLU)
- Dropout Layer
- Dense Layer (ReLU)
- Output Layer (Sigmoid)

Loss Function: Binary Crossentropy  
Optimizer: Adam  
Evaluation Metrics:
- Accuracy
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 📈 Model Performance

- Test Accuracy: 100%
- AUC Score: ~1.00
- No False Positives
- No False Negatives

Note:
Given limited dataset size, further validation using transaction-level fraud data is recommended.

---

## 🚀 Streamlit Dashboard

An interactive dashboard was built to:

- Tune ANN hyperparameters
- Adjust neurons, dropout, learning rate
- Handle class imbalance using class weights
- Visualize confusion matrix & ROC curve
- Dynamically retrain model

### To Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py

# 📊 Vendor Invoice Intelligence Portal

An AI-powered web application that predicts **freight cost** and flags **risky invoices** using machine learning.

This project helps businesses automate invoice validation, detect anomalies, and improve financial decision-making.

---

## 🚀 Project Overview

Manual invoice auditing is slow and error-prone. Industry research shows that a significant percentage of invoices contain errors or inconsistencies. :contentReference[oaicite:1]{index=1}  

This system uses machine learning to:
- Predict freight cost
- Detect abnormal invoices
- Reduce manual review effort
- Improve operational efficiency

---

## 🎯 Features

### 🚚 Freight Cost Prediction
- Predicts freight cost using:
  - Quantity
  - Invoice Dollars
- Helps in budgeting and vendor negotiation

### 🧾 Invoice Risk Detection
- Flags invoices for manual approval
- Detects:
  - Cost mismatch
  - Delay anomalies
- Reduces fraud and incorrect payments

### 🌐 Web Application (Streamlit)
- Interactive UI
- Real-time predictions
- Easy input forms
- Instant results

---

## 🧠 Machine Learning Models

### 1. Freight Cost Model
- Model: Regression Model
- Input Features:
  - Quantity
  - Dollars
- Output:
  - Predicted Freight Cost

---

### 2. Invoice Flag Model
- Model: Random Forest Classifier
- Input Features:
  - invoice_quantity
  - invoice_dollars
  - Freight
  - total_item_quantity
  - total_item_dollars

- Output:
  - 0 → Safe
  - 1 → Flagged (Manual Review Required)

---

## 🏗️ Project Structure

project/
│
├── models/
│ ├── predict_freight_model.pkl
│ ├── predict_flag_invoice.pkl
│ └── scaler.pkl
│
├── inference/
│ ├── predict_freight.py
│ └── predict_invoice_flag.py
│
├── app.py
├── train_flag.py
├── train_freight.py
└── README.md

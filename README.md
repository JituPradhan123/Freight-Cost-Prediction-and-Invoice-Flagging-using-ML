# 📊 Vendor Invoice Intelligence Portal (Freight-Cost-Prediction-and-Invoice-Flagging-using-ML)

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

<img width="1877" height="925" alt="image" src="https://github.com/user-attachments/assets/98a24d6d-144b-4c53-8002-62977ab0d652" />


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

<img width="1886" height="895" alt="image" src="https://github.com/user-attachments/assets/3319a064-7fda-4bf1-b177-06137787ce0d" />




## 🏗️ Project Structure

<img width="232" height="605" alt="image" src="https://github.com/user-attachments/assets/eb41c46b-c0ce-4056-9ed9-6b94659b222f" />


## ▶️ How to Run the Project

Follow these steps to run the application locally.

---

### 1️⃣ Clone the Repository

```bash
https://github.com/JituPradhan123/Freight-Cost-Prediction-and-Invoice-Flagging-using-ML.git
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```
Activate:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### 4️⃣ Train Models (First Time Only)

```
python feright_cost_prediction/train.py
python invoice_flagging/train_flag.py
```

---

### 5️⃣ Run Streamlit App

```
python -m streamlit run app.py
```

👉 Open in browser:

```
http://localhost:8501
```

---

## 🧪 Run Individual Scripts

### Freight Model

```
python inference/predict_freight.py
```

### Invoice Flag Model

```
python inference/predict_invoice_flag.py
```

---

## 🧾 Sample Input

### Freight Prediction

```
{
  "Quantity": [50],
  "Dollars": [5000]
}
```

---

### Invoice Flag Prediction

```
{
  "invoice_quantity": [50],
  "invoice_dollars": [5000],
  "Freight": [200],
  "total_item_quantity": [50],
  "total_item_dollars": [4800]
}
```

---

## 📊 Output

* Freight → Numeric value
* Invoice Flag →

  * ✅ Safe
  * 🚨 Risky

---

## ⚠️ Important Notes

* Use same **scikit-learn version** for training & inference
* Feature names must match exactly
* Always run project from root folder

---

## 🛠️ Troubleshooting

### ❌ Model not found

➡️ Run training scripts again

---

### ❌ sklearn error

```
pip show scikit-learn
```

---

### ❌ Streamlit issue

```
python -m streamlit run app.py
```

---

## 🚀 Future Improvements

* Auto pipeline (no manual freight input)
* Dashboard analytics
* Cloud deployment

---

## 👨‍💻 Author

**Jitu Pradhan**



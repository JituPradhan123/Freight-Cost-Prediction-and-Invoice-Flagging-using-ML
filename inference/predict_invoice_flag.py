import joblib
import pandas as pd

MODEL_PATH = "../models/predict_flag_invoice.pkl"
SCALER_PATH = "../models/scaler.pkl"

# Expected features (IMPORTANT)
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


def load_scaler(scaler_path: str = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        return joblib.load(f)


def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """
    Predict invoice fraud flag (0/1)
    """

    model = load_model()
    scaler = load_scaler()

    df = pd.DataFrame(input_data)

    # ✅ Check missing columns
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ✅ Enforce correct feature order
    df = df[FEATURES]

    # ✅ Scale input
    df_scaled = scaler.transform(df)

    # ✅ Predict
    predictions = model.predict(df_scaled)

    # ✅ Return safe copy
    result = df.copy()
    result["predicted_invoice_flag"] = predictions

    return result


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [20, 56, 30, 30],
        "invoice_dollars": [4765, 5647, 3000, 3000],
        "Freight": [50, 60, 40, 45],
        "total_item_quantity": [20, 55, 28, 29],
        "total_item_dollars": [2000, 5600, 2950, 2980]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)
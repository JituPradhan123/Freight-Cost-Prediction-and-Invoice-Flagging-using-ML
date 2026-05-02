import joblib
import pandas as pd

MODEL_PATH = "../models/predict_freight_model.pkl"

# Expected features (IMPORTANT)
FEATURES = ["Dollars", "Quantity"]


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    """
    Predict freight cost
    """

    model = load_model()
    df_input = pd.DataFrame(input_data)

    # ✅ Check missing columns
    missing = [col for col in FEATURES if col not in df_input.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ✅ Enforce correct order
    df_input = df_input[FEATURES]

    # ✅ Predict
    predictions = model.predict(df_input).round()

    # ✅ Return safe copy
    result = df_input.copy()
    result["predicted_freight_cost"] = predictions

    return result


if __name__ == "__main__":
    sample_data = {
        "Dollars": [4765, 5647, 3000, 3000],
        "Quantity": [20, 56, 30, 30]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
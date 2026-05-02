import joblib
from pathlib import Path
from data_preprossing import (
    load_invoice_data,
    prepare_features,
    split_data
)
from evaluation import (
    train_L_model,
    train_D_model,
    train_R_model,
    evaluate_model
)

def main():
    """
    End-to-end workflow:
    1. Load invoice data
    2. Prepare features and target
    3. Split into train/test sets
    4. Train regression models
    5. Evaluate performance
    6. Save trained models
    """
    db_path = "data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df = load_invoice_data(db_path)
    x, y = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train models
    l_model = train_L_model(x_train, y_train)
    d_model = train_D_model(x_train, y_train)
    r_model = train_R_model(x_train, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model(l_model, x_test, y_test, "Linear Model"))
    results.append(evaluate_model(d_model, x_test, y_test, "Decision Model"))
    results.append(evaluate_model(r_model, x_test, y_test, "Random Model"))

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Model": l_model,
        "Decision Model": d_model,
        "Random Model": r_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()

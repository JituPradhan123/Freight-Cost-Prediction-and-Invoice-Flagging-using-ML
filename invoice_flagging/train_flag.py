from pathlib import Path
import joblib

from data_preprossing_for_flag import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)

from modeling_evaluation_for_flag import (
    train_random_forest,
    evaluate_classifier
)

# Define features and target
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():
    db_path = "../data/inventory.db"

    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)

    # 1. Load data
    df = load_invoice_data(db_path)

    print("Columns:", df.columns)  # Debug check

    # 2. Create labels
    df = apply_labels(df)

    # 3. Split data
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # 4. Scale features
    scaler_path = model_dir / "scaler.pkl"
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, scaler_path
    )

    # 5. Train model
    grid_search = train_random_forest(x_train_scaled, y_train)

    # 6. Evaluate
    evaluate_classifier(
        grid_search.best_estimator_,
        x_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # 7. Save model
    model_path = model_dir / "predict_flag_invoice.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
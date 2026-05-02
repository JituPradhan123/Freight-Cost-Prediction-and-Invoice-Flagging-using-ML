from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer
)


def train_random_forest(x_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    print("Best Params:", grid_search.best_params_)

    return grid_search


def evaluate_classifier(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
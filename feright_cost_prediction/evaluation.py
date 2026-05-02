from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR ###
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error

def train_L_model(x_train,y_train):
    model = LinearRegression()
    model.fit(x_train,y_train)
    return model

def train_D_model(x_train,y_train):
    model = DecisionTreeRegressor(random_state = 42)
    model.fit(x_train,y_train)
    return model

def train_R_model(x_train,y_train):
    model = RandomForestRegressor(random_state = 42)
    model.fit(x_train,y_train)
    return model


def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model and return performance metrics.
    
    Parameters:
        model: Trained regression model
        X_test: Test feature set
        y_test: True target values
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing model name and evaluation metrics
    """
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)*100

    print(f"\n {model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}%")

    # Return dictionary for programmatic use
    return {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_percent": r2
    }

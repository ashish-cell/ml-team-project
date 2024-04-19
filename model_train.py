import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

mlflow.set_tracking_uri("/path/to/new/mlruns")


# Load processed data
data = pd.read_csv('BostonHousing.csv')
X = data.drop('medv', axis=1)
y = data['medv']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLFlow experiment logging
mlflow.set_experiment('boston_housing_prediction')

# Define the model and hyperparameters
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [10, 50, 100],  # Fewer trees for faster execution
    'max_depth': [5, 10, None]
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Start an MLFlow run
with mlflow.start_run():
    mlflow.log_params({"test_size": 0.2, "random_state": 42})

    # Execute the grid search
    grid_search.fit(X_train, y_train)

    # Logging the best parameters and corresponding model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.sklearn.log_model(best_model, "random-forest-regressor")

    # Predict and evaluate the model
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    print(f"Model training completed. Best Parameters: {grid_search.best_params_}")
    print(f"MSE: {mse}")

# Complete the MLFlow logging is handled by the 'with' context
print("MLFlow run completed.")
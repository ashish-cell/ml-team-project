import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

# Load processed data
data = pd.read_csv('BostonHousing.csv')
X = data.drop('medv', axis=1)
y = data['medv']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLFlow: Initialize experiment logging
mlflow.set_experiment('boston_housing_prediction')
mlflow.start_run()
mlflow.log_params({"test_size": 0.2, "random_state": 42})

# Model setup and training
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
mlflow.sklearn.log_model(model, "random-forest-regressor")

# Predict and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mlflow.log_metric("mse", mse)

# Complete the MLFlow logging
mlflow.end_run()

print(f'Model training completed. MSE: {mse}')

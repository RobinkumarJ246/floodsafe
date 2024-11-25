import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Example dataset (replace this with your actual dataset or load from a file)
data = pd.read_csv('evenly_distributed_flood_risk_dataset.csv')

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Split dataset into features (X) and target (y)
X = df[['Rainfall', 'Elevation', 'Impermeability', 'Population']]  # Features
y = df['Risk']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model
xgb_model = xgb.XGBClassifier()

# Hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Set up GridSearchCV with XGBoost
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV on the training data
logging.debug("Starting hyperparameter tuning with GridSearchCV...")
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Log the best parameters
logging.debug(f"Best parameters found: {best_params}")

# Save the best model to a file
joblib.dump(best_model, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
logging.debug("Best model saved as 'best_xgb_model.pkl'.")

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
logging.debug(f"Model accuracy: {best_model.score(X_test_scaled, y_test)}")
logging.debug("Classification report:\n" + classification_report(y_test, y_pred))

# Print the prediction results for the test set
print(f"Test Set Predictions: {y_pred}")
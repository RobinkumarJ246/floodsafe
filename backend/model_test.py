import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the saved model
best_model = joblib.load('best_xgb_model.pkl')
logging.debug("Loaded the best XGBoost model.")

# Load the scaler used for training
scaler = joblib.load('scaler.pkl')  # Ensure to save the scaler during training

# Example input values (Rainfall, Elevation, Impermeability, Population)
# You can replace these with your own test values
example_input = np.array([[100.0, 10.0, 90.0, 120000]])  # Example input

# Scale the input features
scaled_input = scaler.transform(example_input)

# Make a prediction using the trained model
predicted_risk = best_model.predict(scaled_input)

# Output the predicted flood risk
logging.debug(f"Predicted Risk: {predicted_risk[0]}")  # The model returns an array, so we take the first element
print(f"The predicted flood risk for the input data is: {predicted_risk[0]}")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to a list of trusted origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and scaler
try:
    best_model = joblib.load('best_xgb_model.pkl')
    logging.info("Successfully loaded the XGBoost model.")
except Exception as e:
    logging.critical("Failed to load the XGBoost model.", exc_info=True)
    raise

try:
    scaler = joblib.load('scaler.pkl')
    logging.info("Successfully loaded the scaler.")
except Exception as e:
    logging.critical("Failed to load the scaler.", exc_info=True)
    raise

# Define input schema
class InputData(BaseModel):
    rainfall: float
    elevation: float
    impermeability: float
    population: float

# Define the root endpoint
@app.get("/")
def read_root():
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the Flood Risk Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict_risk(data: InputData):
    logging.info(f"Received input data: {data}")
    
    try:
        # Prepare the input for the model
        input_array = np.array([[data.rainfall, data.elevation, data.impermeability, data.population]])
        logging.debug(f"Input array for scaling: {input_array}")
        
        # Scale the input features
        scaled_input = scaler.transform(input_array)
        logging.debug(f"Scaled input: {scaled_input}")
        
        # Predict using the model
        predicted_risk = best_model.predict(scaled_input)
        logging.info(f"Prediction made successfully: {predicted_risk[0]}")
        
        # Convert numpy.int64 to Python int
        python_predicted_risk = int(predicted_risk[0])
        logging.debug(f"Converted prediction to Python int: {python_predicted_risk}")
        
        return {"predicted_risk": python_predicted_risk}
    except Exception as e:
        logging.error("Error during prediction.", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during prediction. Please check the logs.")
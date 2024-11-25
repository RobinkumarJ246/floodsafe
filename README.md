# FloodSafe: Urban Flood Risk Prediction System 🌊🏙️

## Project Overview

FloodSafe is an innovative urban flood risk management platform that integrates cutting-edge geospatial analysis, advanced machine learning, and modern web technologies to provide actionable flood risk insights for urban planners, emergency services, and residents.

## 🎯 Project Motivation

Climate change and urban development have significantly increased flood risks in urban areas. FloodSafe aims to:
- Provide early flood risk assessments
- Support urban planning and infrastructure development
- Enhance community resilience through data-driven insights

## 🌐 Project Repository
Find below the source code of the project in the following github repository

[GitHub Repository: FloodSafe](https://github.com/RobinkumarJ246/floodsafe)

## 🔍 Key Components

### 1. Geospatial Data Extraction (QGIS)

#### Motivation for QGIS Analysis
QGIS was crucial in extracting granular environmental parameters at the ward level. Our spatial analysis focused on two critical metrics:

- **Average Elevation**: Calculated using digital elevation models to understand terrain characteristics
- **Impermeability Rate**: Determined by calculating the ratio of impermeable surfaces (like concrete, asphalt) to the total ward area

##### Detailed Impermeability Rate Calculation
- Overlaid land use datasets with ward boundaries
- Identified impermeable surfaces (buildings, roads, parking lots)
- Computed impermeable area / total ward area
- This metric directly correlates with flood susceptibility, as higher impermeability reduces natural water absorption

### 2. Dataset Preparation

#### Comprehensive Data Sources
- Government environmental datasets
- Public geographical records
- Synthesized data generation using advanced algorithms

#### Dataset Characteristics
- **Total Rows**: Approximately 12,000
- **Features**:
  - Average elevation
  - Population density
  - Impermeability rate
  - Rainfall metrics
- **Target**: Flood risk categories (Very Low to Extreme)

#### Innovative Data Generation Strategy
- Custom algorithm to generate diverse, realistic scenarios
- Ensures model robustness and generalizability
- Simulates multiple urban environmental conditions

##### Dataset generation
```python
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Function to generate evenly distributed flood risk data
def generate_evenly_distributed_flood_risk_data(num_samples_per_class=2000):
    data = []
    
    # Generate samples for each class
    for risk in range(6):
        for _ in range(num_samples_per_class):
            # Randomly generate feature values based on the risk class
            if risk == 5:  # Extreme risk
                rainfall = random.uniform(150, 200)  # High rainfall
                elevation = random.uniform(0, 5)     # Low elevation
                population = random.randint(50000, 200000)  # High population
                impermeability = random.uniform(60, 100)  # High impermeability
            elif risk == 4:  # High risk
                rainfall = random.uniform(100, 150)
                elevation = random.uniform(0, 10)
                population = random.randint(50000, 150000)
                impermeability = random.uniform(50, 80)
            elif risk == 3:  # Moderate risk
                rainfall = random.uniform(70, 100)
                elevation = random.uniform(5, 15)
                population = random.randint(20000, 100000)
                impermeability = random.uniform(30, 60)
            elif risk == 2:  # Low risk
                rainfall = random.uniform(40, 70)
                elevation = random.uniform(10, 18)
                population = random.randint(10000, 100000)
                impermeability = random.uniform(20, 50)
            elif risk == 1:  # Very low risk
                rainfall = random.uniform(20, 40)
                elevation = random.uniform(15, 20)
                population = random.randint(5000, 50000)
                impermeability = random.uniform(10, 30)
            else:  # No risk (very low risk)
                rainfall = random.uniform(0, 20)
                elevation = random.uniform(15, 20)
                population = random.randint(0, 50000)
                impermeability = random.uniform(0, 20)

            data.append([rainfall, elevation, impermeability, population, risk])

    # Convert to a DataFrame
    df = pd.DataFrame(data, columns=["Rainfall", "Elevation", "Impermeability", "Population", "Risk"])

    return df

# Generate the evenly distributed dataset
df = generate_evenly_distributed_flood_risk_data(num_samples_per_class=2000)

# Save the dataset to a CSV file
df.to_csv('evenly_distributed_flood_risk_dataset.csv', index=False)

# Optionally, print the first few rows to verify
print(df.head())

# Split the data into features and target
X = df.drop('Risk', axis=1)  # Features (Rainfall, Elevation, Impermeability, Population)
y = df['Risk']  # Target (Risk)

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Random Forest Classifier for better performance)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importances
features = X.columns
importances = model.feature_importances_

plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Predicting Flood Risk')
plt.show()
```

### 3. Machine Learning Model: XGBoost

#### Why XGBoost?
- **High Prediction Accuracy**: State-of-the-art performance in tabular data prediction
- **Complex Relationship Handling**: Captures non-linear interactions between environmental variables
- **Overfitting Resistance**: Advanced regularization techniques
- **Scalability**: Efficiently processes large, complex datasets

#### Rigorous Model Development
- Comprehensive hyperparameter tuning
- Stratified cross-validation
- Focused on maximizing precision and recall in flood risk prediction
- Ensemble learning approach for enhanced reliability

##### Model training and exporting
```python
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
```

### 4. Backend Architecture: FastAPI

#### Rationale for FastAPI
- **High-Performance**: Asynchronous request handling
- **Automatic API Documentation**: Interactive Swagger UI
- **Machine Learning Integration**: Seamless model deployment
- **Python-Native**: Smooth interaction with data science libraries

##### Backend server
```python
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
```

### 5. Frontend: Next.js

#### Why Next.js?
- **Server-Side Rendering**: Optimized initial page load
- **React Ecosystem**: Modular, component-based architecture
- **Built-in Routing**: Intuitive navigation
- **Production Optimization**: Enhanced user experience

### Homepage
##### Frontend: Home Page
```jsx
'use client';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

const Home = () => {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);

  const navigateToFloodRisk = () => {
    router.push('/wardwise');
  };

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-blue-900 to-indigo-800 text-white">
      {/* Side Menu */}
      <div
        className={`fixed top-0 left-0 h-full w-64 bg-gradient-to-b from-blue-800 to-indigo-700 shadow-lg transition-transform duration-300 z-50 ${
          menuOpen ? 'translate-x-0' : '-translate-x-64'
        }`}
      >
        <div className="flex flex-col h-full">
          {/* Close Button */}
          <button
            onClick={toggleMenu}
            className="text-white text-2xl p-4 self-end focus:outline-none hover:text-gray-300"
          >
            ✕
          </button>

          {/* Profile Section */}
          <div className="flex flex-col items-center mt-6 mb-4">
            <img
              src="https://avatar.iran.liara.run/public/20" // Replace with actual profile image URL
              alt="User Profile"
              className="w-20 h-20 rounded-full border-4 border-white shadow-lg"
            />
            <h2 className="mt-3 text-lg font-bold">Magesh S</h2>
            <p className="text-sm text-gray-300">mageshss@gmail.com</p>
          </div>

          {/* Menu Items */}
          <div className="flex-1">
            <nav className="mt-4 space-y-2">
              <button
                onClick={() => router.push('/prediction')}
                className="w-full text-left px-6 py-3 text-white hover:bg-blue-600 rounded-md transition"
              >
                📊 Dashboard
              </button>
              <button
                onClick={() => router.push('/settings')}
                className="w-full text-left px-6 py-3 text-white hover:bg-blue-600 rounded-md transition"
              >
                ⚙️ Settings
              </button>
              <button
                onClick={() => router.push('/help')}
                className="w-full text-left px-6 py-3 text-white hover:bg-blue-600 rounded-md transition"
              >
                ❔ Help
              </button>
              <button
                onClick={() => router.push('/switch-account')}
                className="w-full text-left px-6 py-3 text-white hover:bg-blue-600 rounded-md transition"
              >
                🔄 Switch Account
              </button>
              <button
                onClick={() => router.push('/logout')}
                className="w-full text-left px-6 py-3 text-white hover:bg-red-600 rounded-md transition"
              >
                🚪 Logout
              </button>
            </nav>
          </div>

          {/* Footer Section in Sidebar */}
          <footer className="text-center mt-auto p-4 text-sm text-gray-300">
            © 2024 FloodSafe
          </footer>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header Section */}
        <header className="flex items-center justify-between px-6 py-4 bg-blue-900 shadow-md">
          <button
            onClick={toggleMenu}
            className="text-2xl text-white focus:outline-none"
          >
            ☰
          </button>
          <h1 className="text-3xl font-bold text-center flex-grow">
            FloodSafe
          </h1>
        </header>

        {/* Hero Section */}
        <div className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="text-center">
            <h1 className="text-5xl font-bold mb-6">
              Stay Ahead of the Flood
            </h1>
            <p className="text-lg max-w-3xl mx-auto mb-10">
              FloodSafe provides real-time flood risk analysis, predictive
              models, and a comprehensive dashboard to help you make informed
              decisions and stay prepared.
            </p>
            <button
              onClick={navigateToFloodRisk}
              className="px-8 py-4 bg-indigo-600 hover:bg-indigo-700 text-lg font-semibold rounded-lg shadow-lg transition"
            >
              🌊 Get Started with Flood Prediction
            </button>
          </div>
        </div>

        {/* Footer Section */}
        <footer className="bg-blue-900 text-white text-center py-4">
          <p className="text-sm">
            © 2024 FloodSafe | Made for smarter flood management
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Home;
```


### Wardwise
##### Frontend: Flood Risk Prediction Page Wardwise
```jsx
'use client';
import { useState } from "react";
import wardwiseData from './wardwise_data.json'; // Import the JSON file

const FloodRiskByWard = () => {
  const [ward, setWard] = useState("");
  const [wardDetails, setWardDetails] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [reasoning, setReasoning] = useState("");
  const [additionalInfo, setAdditionalInfo] = useState("");
  const [errors, setErrors] = useState("");
  const [rainfall, setRainfall] = useState(""); // New input for user-provided rainfall data

  // Flood risk level mapping
  const riskLevels = [
    "Very Low",           // Risk level 0
    "Low",           // Risk level 1
    "Moderate",      // Risk level 2
    "High",          // Risk level 3
    "Very High",     // Risk level 4
    "Extreme"        // Risk level 5
  ];

  // Define colors for each risk level
  const riskColors = {
    Low: "text-green-500",
    Moderate: "text-yellow-500",
    High: "text-orange-500",
    "Very High": "text-red-500",
    Extreme: "text-red-800",
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors("");
    setPrediction(null);
    setReasoning("");
    setAdditionalInfo("");

    // Validate ward input
    const wardData = wardwiseData.find((item) => item.ward === parseInt(ward));
    if (!wardData) {
      setErrors("Invalid ward number. Please enter a valid ward.");
      return;
    }
    if (!rainfall || isNaN(rainfall) || parseFloat(rainfall) <= 0) {
      setErrors("Please enter a valid rainfall value (mm).");
      return;
    }

    setWardDetails(wardData); // Save ward details for display
    const { avg_elev, imper_rate, population } = wardData;

    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          elevation: avg_elev,
          rainfall: parseFloat(rainfall),
          population,
          impermeability: imper_rate,
        }),
      });

      if (!response.ok) {
        throw new Error("Error in prediction request");
      }

      const data = await response.json();
      const riskLevel = data.predicted_risk;
      setPrediction(riskLevels[riskLevel]);
      generateReasoning(
        riskLevel,
        parseFloat(rainfall),
        parseFloat(population),
        parseInt(imper_rate),
        parseFloat(avg_elev)
      );
    } catch (error) {
      setErrors("An error occurred while fetching the prediction.");
      console.error("Error:", error);
    }
  };

  const generateReasoning = (
    riskPrediction,
    rainfall,
    population,
    impermeabilityIndex,
    elevation
  ) => {
    let explanation = "";

    const elevationEffect = elevation < 10 ? "low elevation area" : "higher elevation area";
    const rainfallEffect = rainfall > 200
      ? "extremely high rainfall"
      : rainfall > 100
      ? "heavy rainfall"
      : "moderate rainfall";
    const populationEffect = population > 1000 ? "high population" : "low population";
    const impermeabilityEffect =
      impermeabilityIndex > 70 ? "high impermeability" : impermeabilityIndex > 40 ? "moderate impermeability" : "low impermeability";

    if (riskPrediction === 0) {
      explanation = `The area has a very low flood risk due to ${elevationEffect}, ${impermeabilityEffect}, and ${populationEffect}.`;
    } else if (riskPrediction === 1) {
      explanation = `The flood risk is low, primarily due to ${impermeabilityEffect} and ${elevationEffect}.`;
    } else if (riskPrediction === 2) {
      explanation = `The risk is moderate due to ${rainfallEffect}, ${populationEffect}, and ${impermeabilityEffect}.`;
    } else if (riskPrediction === 3) {
      explanation = `High flood risk driven by ${rainfallEffect} and ${populationEffect}, exacerbated by ${impermeabilityEffect}.`;
    } else if (riskPrediction === 4) {
      explanation = `The area faces a very high flood risk due to ${rainfallEffect} and ${impermeabilityEffect}.`;
    } else if (riskPrediction === 5) {
      explanation = `Extreme flood risk due to ${rainfallEffect}, ${impermeabilityEffect}, and ${populationEffect}.`;
    } else {
      explanation = "Unable to determine reasoning for this risk level.";
    }

    setReasoning(explanation);
    setAdditionalInfo(
      generateAdditionalInfo(riskPrediction, rainfall, population, impermeabilityIndex, elevation)
    );
  };

  const generateAdditionalInfo = (
    riskPrediction,
    rainfall,
    population,
    impermeabilityIndex,
    elevation
  ) => {
    let info = "";

    if (riskPrediction > 2 && population > 50000) {
      info += "This area has a higher population, which impacts the severity and potential damage of floods. " +
              "Effective evacuation plans and infrastructure improvements are critical to minimize the risk to the population. ";
    }

    if (elevation < 2) {
      info += "This area is at a low elevation, which makes it more prone to flooding and causes water to stagnate easily, worsening the situation. ";
    }

    if (impermeabilityIndex > 90) {
      info += "The impermeability rate is high, meaning the area is less capable of draining water. " +
              "This increases the risk of flooding as the water cannot be absorbed into the ground. ";
    }

    if (rainfall > 180) {
      info += "The area is experiencing unusually high rainfall, which could overwhelm drainage systems and increase flood risk. ";
    }

    return info;
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-blue-500 via-green-600 to-blue-800 flex items-center justify-center p-6">
      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-4xl">
        <h1 className="text-4xl font-bold text-indigo-800 text-center mb-6">Flood Risk by Ward</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Ward Number</label>
            <input
              type="number"
              value={ward}
              onChange={(e) => setWard(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter ward number"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Rainfall (mm)</label>
            <input
              type="number"
              value={rainfall}
              onChange={(e) => setRainfall(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter rainfall value in mm"
            />
          </div>
          {errors && <span className="text-red-500 text-sm">{errors}</span>}
          <div className="text-center">
            <button
              type="submit"
              className="bg-violet-700 text-white font-bold p-4 rounded-xl w-full max-w-xs hover:bg-violet-800"
            >
              🔍 Analyze flood risk
            </button>
          </div>
        </form>
        {wardDetails && (
          <div className="mt-8 bg-gray-100 p-4 rounded-lg">
            <h2 className="text-xl font-bold">Ward Details</h2>
            <p><strong>Ward Number:</strong> {wardDetails.ward}</p>
            <p><strong>Zone:</strong> {wardDetails.zone}</p>
            <p><strong>Population:</strong> {wardDetails.population}</p>
            <p><strong>Elevation:</strong> {wardDetails.avg_elev} meters</p>
            <p><strong>Total Area:</strong> {wardDetails.area_km} km²</p>
            <p><strong>Total Impermeable Area:</strong> {wardDetails.reproj_dissolved_buildings_wards_imper_km} km²</p>
          </div>
        )}
        {prediction !== null && (
          <div className="mt-8 text-center">
            <div className={`text-3xl font-semibold ${riskColors[prediction]}`}>
              Predicted Risk Level: {prediction}
            </div>
            <p className="mt-4">{reasoning}</p>
            <div className="text-gray-700 mt-4">{additionalInfo}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FloodRiskByWard;
```

### Manual data entry

##### Frontend: Flood Risk Prediction Manual Data Entry
```jsx
'use client';
import { useState } from "react";

const FloodRiskPrediction = () => {
  const [elevation, setElevation] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [population, setpopulation] = useState("");
  const [impermeabilityIndex, setImpermeabilityIndex] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [reasoning, setReasoning] = useState("");
  const [additionalInfo, setAdditionalInfo] = useState("");
  const [errors, setErrors] = useState({
    elevation: "",
    rainfall: "",
    population: "",
    impermeabilityIndex: "",
  });

  const validRanges = {
    elevation: { min: 0, max: 20 },
    rainfall: { min: 0, max: 200 },
    population: { min: 0, max: 200000 },
    impermeabilityIndex: { min: 0, max: 100 },
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors({
      elevation: "",
      rainfall: "",
      population: "",
      impermeabilityIndex: "",
    });
    const newErrors = {};

    if (!validateInput(elevation, validRanges.elevation, "elevation")) {
      newErrors.elevation = `Elevation must be between ${validRanges.elevation.min} and ${validRanges.elevation.max} meters.`;
    }
    if (!validateInput(rainfall, validRanges.rainfall, "rainfall")) {
      newErrors.rainfall = `Rainfall must be between ${validRanges.rainfall.min} and ${validRanges.rainfall.max} mm.`;
    }
    if (!validateInput(population, validRanges.population, "population")) {
      newErrors.population = `Population must be between ${validRanges.population.min} and ${validRanges.population.max}.`;
    }
    if (!validateInput(impermeabilityIndex, validRanges.impermeabilityIndex, "impermeabilityIndex")) {
      newErrors.impermeabilityIndex = `Impermeability index must be between ${validRanges.impermeabilityIndex.min} and ${validRanges.impermeabilityIndex.max}.`;
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          elevation: parseFloat(elevation),
          rainfall: parseFloat(rainfall),
          population: parseFloat(population),
          impermeability: parseInt(impermeabilityIndex),
        }),
      });

      if (!response.ok) {
        console.error("Error in prediction request");
        return;
      }

      const data = await response.json();
      setPrediction(data.predicted_risk);
      generateReasoning(
        data.predicted_risk,
        parseFloat(rainfall),
        parseFloat(population),
        parseInt(impermeabilityIndex),
        parseFloat(elevation)
      );
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const validateInput = (value, range, field) => {
    return value >= range.min && value <= range.max;
  };

  const generateReasoning = (
    riskPrediction,
    rainfall,
    population,
    impermeabilityIndex,
    elevation
  ) => {
    let explanation = "";

    const elevationEffect = elevation < 10 ? "low elevation area" : "higher elevation area";
    const rainfallEffect = rainfall > 200
      ? "extremely high rainfall"
      : rainfall > 100
      ? "heavy rainfall"
      : "moderate rainfall";
    const populationEffect = population > 1000 ? "high population" : "low population";
    const impermeabilityEffect =
      impermeabilityIndex > 70 ? "high impermeability" : impermeabilityIndex > 40 ? "moderate impermeability" : "low impermeability";

    if (riskPrediction === 0) {
      explanation = `The area has a very low flood risk due to ${elevationEffect}, ${impermeabilityEffect}, and ${populationEffect}.`;
    } else if (riskPrediction === 1) {
      explanation = `The flood risk is low, primarily due to ${impermeabilityEffect} and ${elevationEffect}.`;
    } else if (riskPrediction === 2) {
      explanation = `The risk is moderate due to ${rainfallEffect}, ${populationEffect}, and ${impermeabilityEffect}.`;
    } else if (riskPrediction === 3) {
      explanation = `High flood risk driven by ${rainfallEffect} and ${populationEffect}, exacerbated by ${impermeabilityEffect}.`;
    } else if (riskPrediction === 4) {
      explanation = `The area faces a very high flood risk due to ${rainfallEffect} and ${impermeabilityEffect}.`;
    } else if (riskPrediction === 5) {
      explanation = `Extreme flood risk due to ${rainfallEffect}, ${impermeabilityEffect}, and ${populationEffect}.`;
    } else {
      explanation = "Unable to determine reasoning for this risk level.";
    }

    setReasoning(explanation);
    setAdditionalInfo(generateAdditionalInfo(riskPrediction, rainfall, population, impermeabilityIndex, elevation));
  };

  const generateAdditionalInfo = (
    riskPrediction,
    rainfall,
    population,
    impermeabilityIndex,
    elevation
  ) => {
    let info = "";

    if (riskPrediction > 2 && population > 50000) {
      info += "This area has a higher population, which impacts the severity and potential damage of floods. " +
              "Effective evacuation plans and infrastructure improvements are critical to minimize the risk to the population. ";
    }

    if (elevation < 2) {
      info += "This area is at a low elevation, which makes it more prone to flooding and causes water to stagnate easily, worsening the situation. ";
    }

    if (impermeabilityIndex > 90) {
      info += "The impermeability rate is high, meaning the area is less capable of draining water. " +
              "This increases the risk of flooding as the water cannot be absorbed into the ground. ";
    }

    if (rainfall > 180) {
      info += "The area is experiencing unusually high rainfall, which could overwhelm drainage systems and increase flood risk. ";
    }

    return info;
  };

  const mapPredictionToLabel = (prediction) => {
    switch (prediction) {
      case 0:
        return { label: "Very Low", color: "text-green-500", barColor: "bg-green-300" };
      case 1:
        return { label: "Low", color: "text-blue-500", barColor: "bg-blue-300" };
      case 2:
        return { label: "Moderate", color: "text-yellow-500", barColor: "bg-yellow-300" };
      case 3:
        return { label: "High", color: "text-orange-500", barColor: "bg-orange-300" };
      case 4:
        return { label: "Very High", color: "text-red-500", barColor: "bg-red-300" };
      case 5:
        return { label: "Extreme", color: "text-red-800", barColor: "bg-red-700" };
      default:
        return { label: "Unknown", color: "text-gray-500", barColor: "bg-gray-300" };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-blue-500 via-green-600 to-blue-800 flex items-center justify-center p-6">
      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-indigo-800">Floodsafe: Chennai Flood Risk Analysis</h1>
          <p className="mt-4 text-xl text-gray-700">
            Floodsafe is a mini project by the batch 2024-2025 of Sri Venkateswara College of Engineering, aiming to analyze flood risk in the Chennai region using data such as elevation, population, and impermeability rate of wards. We utilize QGIS software for ward-wise analysis to accurately predict flood risk.
          </p>
        </div>

        <div className="mt-8 text-center text-gray-700 mb-8">
          <h3 className="text-2xl font-semibold text-indigo-800">Meet Our Team</h3>
          <p className="mt-2 text-lg text-gray-600">The brilliant minds behind Floodsafe: Chennai Flood Risk Analysis</p>
          <div className="mt-6 flex justify-center space-x-12">
            {/* Team Member 1 */}
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-indigo-500 text-white p-4 rounded-full shadow-xl">
                <span className="text-3xl font-semibold">⭐</span>
              </div>
              <span className="font-semibold text-lg text-indigo-800">Kiran Sekar S</span>
              <p className="text-gray-600">Team Lead</p>
            </div>
            {/* Team Member 2 */}
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-indigo-500 text-white p-4 rounded-full shadow-xl">
                <span className="text-3xl font-semibold">⭐</span>
              </div>
              <span className="font-semibold text-lg text-indigo-800">Magesh S</span>
              <p className="text-gray-600">Data Scientist</p>
            </div>
            {/* Team Member 3 */}
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-indigo-500 text-white p-4 rounded-full shadow-xl">
                <span className="text-3xl font-semibold">⭐</span>
              </div>
              <span className="font-semibold text-lg text-indigo-800">Robinkumar J</span>
              <p className="text-gray-600">Developer</p>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Input Fields */}
          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Elevation (meters)</label>
            <input
              type="number"
              value={elevation}
              onChange={(e) => setElevation(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter elevation"
            />
            {errors.elevation && <span className="text-red-500 text-sm">{errors.elevation}</span>}
          </div>

          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Rainfall (mm)</label>
            <input
              type="number"
              value={rainfall}
              onChange={(e) => setRainfall(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter rainfall"
            />
            {errors.rainfall && <span className="text-red-500 text-sm">{errors.rainfall}</span>}
          </div>

          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Population</label>
            <input
              type="number"
              value={population}
              onChange={(e) => setpopulation(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter population"
            />
            {errors.population && <span className="text-red-500 text-sm">{errors.population}</span>}
          </div>

          <div className="flex flex-col">
            <label className="text-xl text-gray-700">Impermeability Index</label>
            <input
              type="number"
              value={impermeabilityIndex}
              onChange={(e) => setImpermeabilityIndex(e.target.value)}
              className="border-2 rounded-lg p-2 mt-2"
              placeholder="Enter impermeability index"
            />
            {errors.impermeabilityIndex && <span className="text-red-500 text-sm">{errors.impermeabilityIndex}</span>}
          </div>

          <div className="text-center">
            <button
              type="submit"
              className="bg-violet-700 text-white font-bold p-4 rounded-xl w-full max-w-xs hover:bg-violet-800"
            >
              🔍 Analyze flood risk
            </button>
          </div>
        </form>

        {prediction !== null && (
  <div className="mt-8 text-center">
    <div
      className={`text-3xl font-semibold ${mapPredictionToLabel(prediction).color}`}
    >
      {mapPredictionToLabel(prediction).label}
    </div>
    <div
      className={`w-full h-4 mt-2 rounded-full ${mapPredictionToLabel(prediction).barColor}`}
    ></div>
    <p className="mt-4 text-xl font-semibold">What does this mean?</p>
    <p className="mt-2 text-xl">{reasoning}</p>

    {additionalInfo && additionalInfo.length > 0 && (
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-2 gap-4">
        {additionalInfo.split(". ").map((info, index) => (
          info.trim() && (
            <div
              key={index}
              className="bg-gray-100 p-4 rounded-lg shadow-md border-2 border-red-500"
            >
              <p className="text-lg text-gray-700">{info}</p>
            </div>
          )
        ))}
      </div>
    )}
  </div>
)}


      </div>
    </div>
  );
};

export default FloodRiskPrediction;
```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- pip 23.1.2
- NextJS v15.0.3
- Node.js 14+
- yarn v1.22.22
- QGIS v3.34.9-Prizren

### Clone the Repository
```bash
git clone https://github.com/RobinkumarJ246/floodsafe.git
cd floodsafe
```

### Backend Setup (Python)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run model training
python model.py

# Start FastAPI server
uvicorn main:app --reload
```

### Frontend Setup (Next.js)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## 🔬 Running the Application
1. Start the FastAPI backend
2. Launch the Next.js frontend
3. Access the application at `http://localhost:3000`

## 🛠 System Workflow
1. Geospatial data extraction in QGIS
2. Dataset preparation and augmentation
3. XGBoost model training
4. Model serialization
5. FastAPI backend development
6. Next.js frontend implementation

## 🚧 Future Enhancements
- Real-time data integration
- Advanced risk stratification
- Climate change scenario modeling
- Machine learning model versioning
- Extended geographical coverage

## 🛡️ Technologies
- QGIS
- Python
- XGBoost
- FastAPI
- Next.js
- Scikit-learn
- Joblib
- Docker (optional containerization)

## 🤝 Contributors
- Robinkumar J
- Magesh S
- KiranSekar S

## 🏆 Conclusion
FloodSafe represents a holistic approach to urban flood risk management, seamlessly bridging geospatial analysis, machine learning, and web technologies to create actionable, data-driven insights.
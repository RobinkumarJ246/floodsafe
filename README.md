# Urban Flood Risk Prediction System

## Project Overview

This project develops a comprehensive flood risk prediction system that integrates geospatial analysis, machine learning, and web technologies to provide actionable insights for urban flood management.

## Key Components

### 1. Geospatial Data Extraction (QGIS)

#### Motivation for QGIS Analysis
QGIS was crucial in extracting granular environmental parameters at the ward level. Our spatial analysis focused on two critical metrics:

- **Average Elevation**: Calculated using digital elevation models to understand terrain characteristics
- **Impermeability Rate**: Determined by calculating the ratio of impermeable surfaces (like concrete, asphalt) to the total ward area

##### Impermeability Rate Calculation
- Overlaid land use datasets with ward boundaries
- Identified impermeable surfaces (buildings, roads, parking lots)
- Computed impermeable area / total ward area
- This metric directly correlates with flood susceptibility, as higher impermeability reduces natural water absorption

### 2. Dataset Preparation

#### Data Sources
- Government datasets
- Public environmental records
- Synthesized data generation

#### Dataset Characteristics
- Total rows: Approximately 10,000
- Features:
  - Average elevation
  - Population density
  - Impermeability rate
  - Rainfall metrics
- Target: Flood risk categories (Very Low to Extreme)

#### Data Generation Strategy
Used a custom algorithm to generate diverse, realistic scenarios, ensuring model robustness and generalizability.

### 3. Machine Learning Model: XGBoost

#### Why XGBoost?
- **High Prediction Accuracy**: Excels in tabular data prediction
- **Efficient Handling of Complex Relationships**: Can capture non-linear interactions between environmental variables
- **Robust to Overfitting**: Built-in regularization techniques
- **Scalability**: Handles large datasets efficiently

#### Model Development
- Comprehensive hyperparameter tuning
- Cross-validation for performance optimization
- Focused on maximizing precision and recall in flood risk prediction

##### Placeholder: Model Training Code
```python
# Placeholder for XGBoost model training script
def train_xgboost_model(X_train, y_train):
    # Hyperparameter tuning
    # Model training
    # Evaluation and serialization
    pass
```

### 4. Backend Architecture: FastAPI

#### Rationale for FastAPI
- **High Performance**: Asynchronous request handling
- **Automatic API Documentation**: Swagger UI integration
- **Easy Integration**: Seamless machine learning model deployment
- **Python-Native**: Smooth interaction with scikit-learn and XGBoost

##### Placeholder: FastAPI Server Code
```python
# Placeholder for FastAPI server implementation
@app.post("/predict-flood-risk")
async def predict_flood_risk(input_data: InputModel):
    # Load serialized XGBoost model
    # Process input
    # Return predictions and recommendations
    pass
```

### 5. Frontend: Next.js

#### Why Next.js?
- **Server-Side Rendering**: Improved initial load performance
- **React Ecosystem**: Component-based architecture
- **Built-in Routing**: Simplified navigation
- **Optimized Production Build**: Enhanced user experience

##### Placeholder: Frontend Components
```jsx
// Placeholder for risk prediction page component
function RiskPredictionPage() {
  // Input form
  // Dynamic visualization
  // Recommendation display
}
```

### 6. System Workflow
1. Geospatial data extraction in QGIS
2. Dataset preparation and augmentation
3. XGBoost model training
4. Model serialization
5. FastAPI backend development
6. Next.js frontend implementation

## Future Enhancements
- Real-time data integration
- More granular risk stratification
- Climate change scenario modeling

## Technologies
- QGIS
- Python
- XGBoost
- FastAPI
- Next.js
- Scikit-learn
- Joblib

## Conclusion
A holistic approach to urban flood risk management, bridging geospatial analysis, machine learning, and web technologies.
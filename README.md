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
- **Total Rows**: Approximately 10,000
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

### 4. Backend Architecture: FastAPI

#### Rationale for FastAPI
- **High-Performance**: Asynchronous request handling
- **Automatic API Documentation**: Interactive Swagger UI
- **Machine Learning Integration**: Seamless model deployment
- **Python-Native**: Smooth interaction with data science libraries

### 5. Frontend: Next.js

#### Why Next.js?
- **Server-Side Rendering**: Optimized initial page load
- **React Ecosystem**: Modular, component-based architecture
- **Built-in Routing**: Intuitive navigation
- **Production Optimization**: Enhanced user experience

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn
- QGIS (latest version)

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
python train_model.py

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

### QGIS Data Preparation
1. Install QGIS
2. Load ward boundary datasets
3. Process elevation and land use data
4. Export preprocessed geospatial data

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
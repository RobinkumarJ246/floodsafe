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
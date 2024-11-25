import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the model and dataset
flood_model = joblib.load('flood_risk_model.pkl')
df = pd.read_csv('flood_risk_dataset_new.csv')

# Prepare features and target
X = df[['Rainfall', 'Elevation', 'Population', 'Impermeability']]
y = df['Risk']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on the test set
y_pred = flood_model.predict(X_test)

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'], yticklabels=['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
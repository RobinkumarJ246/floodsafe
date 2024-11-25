import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('flood_risk_dataset_new.csv')

# Summary statistics for all features
print("Dataset Summary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Check the distribution of the target variable 'Risk'
print("\nDistribution of Risk Levels:")
print(df['Risk'].value_counts())

# Plot the distribution of each feature
features = ['Rainfall', 'Elevation', 'Population', 'Impermeability']
df[features].hist(bins=20, figsize=(12, 8))
plt.suptitle('Feature Distributions')
plt.show()

# Visualize the relationship between features and target
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Risk', y=feature, data=df)
    plt.title(f'Distribution of {feature} by Risk Level')
    plt.show()
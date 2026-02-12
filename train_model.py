import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('transactions.csv')

# Create target
df['IsFraud'] = df['Status'].apply(lambda x: 1 if x == 'FAILED' else 0)

# Keep only useful feature
df = df[['Amount (INR)', 'IsFraud']]

X = df[['Amount (INR)']]
y = df['IsFraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (NO SCALER NEEDED)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and feature names
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ Model trained and saved successfully")

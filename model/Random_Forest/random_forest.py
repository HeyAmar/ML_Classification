import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Load dataset
df = pd.read_csv('DataSet/heart_disease_uci.csv')

# Drop id column if present
if 'id' in df.columns:
    df = df.drop('id', axis=1)
# Encode categorical columns
for col in df.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
# Handle missing values (simple fillna, can be improved)
df = df.fillna(df.median(numeric_only=True))
X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

joblib.dump(model, f"model/RandomForestClassifier.pkl")
print(f"RandomForestClassifier.pkl saved")


print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

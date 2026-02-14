import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("DataSet/heart_disease_uci.csv")

X = df.drop("num", axis=1)
y = df["num"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Create Model Folder
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# Train Models
# -----------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")
    print(f"{name}.pkl saved")

# -----------------------------
# Save Preprocessing Objects
# -----------------------------
joblib.dump(imputer, "model/imputer.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("All models and preprocessors saved successfully!")

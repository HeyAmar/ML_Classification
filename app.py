
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


st.title("Heart Disease Multi-Class Classification")

# =========================
# Download test.csv
# =========================
st.subheader("Download Test Dataset")

try:
    with open("test.csv", "rb") as file:
        st.download_button(
            label="Download test.csv",
            data=file,
            file_name="test.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("test.csv not found in project root folder.")


# =========================
# Upload CSV
# =========================
st.subheader("Upload Test Dataset")
uploaded_file = st.file_uploader("Upload test.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload test.csv to run prediction.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Uploaded Data Preview")
st.dataframe(df.head())


# =========================
# Validate dataset
# =========================
if "target" not in df.columns:
    st.error("Dataset must contain target column 'num'.")
    st.stop()

X = df.drop("target", axis=1)
y_true = df["target"]

# =========================
# Model selection
# =========================
model_files = {
    "Logistic Regression": "model/Logistic_Regression/LogisticRegression.pkl",
    "Decision Tree": "model/Decision_Tree_Classifier/decision_tree_classifier.pkl",
    "KNN": "model/K_Nearest_Neighbor_Classifier/KNeighborsClassifier.pkl",
    "Naive Bayes": "model/Naive_Bayes_Classifier/naive_bayes_classifier.pkl",
    "Random Forest": "model/Random_Forest/RandomForestClassifier.pkl",
    "XGBoost": "model/XGBoost/XGBClassifier.pkl"
}


model_name = st.selectbox("Select Model", list(model_files.keys()))
model = joblib.load(model_files[model_name])

y_pred = model.predict(X)

	# Try to get probability for AUC

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
mcc = matthews_corrcoef(y_true, y_pred)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
else:
    auc = None

	# c. Display of evaluation metrics

    
st.subheader("Evaluation Metrics")

st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"AUC Score: {auc:.2f}" if auc else "AUC Score: N/A")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")
st.write(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

st.write("## Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

st.write("## Classification Report")
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# st.subheader("Classification Report")
st.dataframe(
    report_df.style
    .format("{:.3f}")
    .background_gradient(cmap="Blues")
)

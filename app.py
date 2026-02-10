
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


st.title("ML Classification App")

# Load dataset from file
@st.cache_data
def load_data():
	df = pd.read_csv("DataSet/heart_disease_uci.csv")
	return df

df = load_data()
st.write("### Heart Disease Dataset Preview", df.head())

# Exploratory Data Analysis
st.write("## Exploratory Data Analysis")
st.write("**Shape of dataset:**", df.shape)
st.write("**Columns:**", df.columns.tolist())
st.write("**Missing values:**")
st.write(df.isnull().sum())
st.write("**Data types:**")
st.write(df.dtypes)
st.write("**Target value counts:**")
st.write(df['num'].value_counts())

# Show basic statistics
st.write("**Summary statistics:**")
st.write(df.describe(include='all'))

# Show correlation heatmap
st.write("**Correlation Heatmap:**")
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# a. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload your test dataset (CSV only)", type=["csv"])
df = None
if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	st.write("### Uploaded Data Preview", df.head())

# b. Model selection dropdown
model_options = {
	"Logistic Regression": LogisticRegression(max_iter=1000),
	"Decision Tree": DecisionTreeClassifier(random_state=42),
	"K-Nearest Neighbor": KNeighborsClassifier(),
	"Naive Bayes": GaussianNB(),
	"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
	"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}
model_name = st.selectbox("Select Model", list(model_options.keys()))
model = model_options[model_name]

# Only proceed if data is uploaded
if df is not None:
	# Assume last column is the target
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	# Try to get probability for AUC
	if hasattr(model, "predict_proba"):
		try:
			y_proba = model.predict_proba(X_test)
			# If binary, use column 1; if multiclass, use macro average
			if y_proba.shape[1] == 2:
				auc = roc_auc_score(y_test, y_proba[:, 1])
			else:
				auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
		except Exception:
			auc = None
	else:
		auc = None

	mcc = matthews_corrcoef(y_test, y_pred)

	# c. Display of evaluation metrics
	st.write("## Evaluation Metrics")
	st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
	st.write(f"AUC Score: {auc:.2f}" if auc is not None else "AUC Score: Not available")
	st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
	st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
	st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
	st.write(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")

	# d. Confusion matrix or classification report
	st.write("## Confusion Matrix")
	cm = confusion_matrix(y_test, y_pred)
	fig, ax = plt.subplots()
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
	st.pyplot(fig)

	st.write("## Classification Report")
	st.text(classification_report(y_test, y_pred))

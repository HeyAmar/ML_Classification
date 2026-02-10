
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Classification App")

# a. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload your test dataset (CSV only)", type=["csv"])
df = None
if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	st.write("### Uploaded Data Preview", df.head())

# b. Model selection dropdown
model_options = {
	"Random Forest": RandomForestClassifier(),
	"Logistic Regression": LogisticRegression(max_iter=1000)
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

	# c. Display of evaluation metrics
	st.write("## Evaluation Metrics")
	st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
	st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
	st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
	st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

	# d. Confusion matrix or classification report
	st.write("## Confusion Matrix")
	cm = confusion_matrix(y_test, y_pred)
	fig, ax = plt.subplots()
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
	st.pyplot(fig)

	st.write("## Classification Report")
	st.text(classification_report(y_test, y_pred))

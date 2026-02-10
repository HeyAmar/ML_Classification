import streamlit as st
import pandas as pd
from model.data_loader import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Streamlit file uploader
df = None
uploaded_file = st.file_uploader("Upload your test dataset (CSV only)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Save uploaded file to a temp location for loader
    temp_path = "temp_uploaded.csv"
    df.to_csv(temp_path, index=False)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(temp_path)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        st.write(f"## {name}")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                st.write(f"AUC Score: {auc:.2f}")
            except Exception:
                st.write("AUC Score: Not available")
        else:
            st.write("AUC Score: Not available")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

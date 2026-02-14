# ML Classification 

## Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early prediction using clinical parameters can support timely medical intervention.

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease using patient medical attributes. The trained models are integrated into an interactive Streamlit web application for prediction and evaluation.

## Dataset Description

This project uses the Heart Disease dataset obtained from the UCI Machine Learning Repository.

Dataset Characteristics

Total Instances: ~920

Number of Features: 13 clinical attributes

Target Variable: num

Problem Type: Binary Classification

Target Variable

The target variable num represents the presence of heart disease:

Class	Interpretation
0	No heart disease
1	Mild heart disease
2	Moderate heart disease
3	Severe heart disease
4	Very severe heart disease

The original dataset contains severity levels (0–4). For this study, values greater than 0 are mapped to 1 to create a binary classification problem.

Example Features

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol level

Fasting blood sugar

Maximum heart rate achieved

Exercise induced angina

Missing values are handled using median imputation during preprocessing.

## Models Used

| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC |
|---------------------|----------|------|-----------|--------|------|-----|
| Logistic Regression | 0.59     | 0.76 | 0.61      | 0.59   | 0.59 | 0.37 |
| Decision Tree       | 0.57     | 0.65 | 0.60      | 0.57   | 0.58 | 0.35 |
| kNN                 | 0.46     | 0.63 | 0.45      | 0.46   | 0.45 | 0.13 |
| Naive Bayes         | 0.49     | 0.64 | 0.62      | 0.49   | 0.54 | 0.28 |
| Random Forest       | 0.57     | 0.78 | 0.54      | 0.57   | 0.55 | 0.32 |
| XGBoost             | 0.62     | 0.85 | 0.61      | 0.62   | 0.62 | 0.40 |


## Model Performance Observations
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Provides strong baseline performance with balanced precision and recall. It models linear relationships effectively and achieves good AUC and MCC, indicating stable multi-class prediction. |
| Decision Tree       | Captures non-linear patterns but shows slightly lower AUC than Logistic Regression. Performance is moderate due to tendency to overfit training data. |
| kNN                 | Shows the lowest overall performance. The model is sensitive to feature scaling and data noise, resulting in low accuracy and MCC. |
| Naive Bayes         | Achieves moderate performance with relatively high precision but lower recall. The independence assumption between features limits predictive capability. |
| Random Forest       | Performs better than a single decision tree due to ensemble learning. It provides stable metrics and improved robustness with good AUC. |
| XGBoost             | Delivers the best overall performance across all metrics. High AUC, F1, and MCC indicate strong ability to model complex relationships and class separation. |

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Deployment

- [Streamlit App Link](#)
- [GitHub Repository Link] - https://github.com/HeyAmar/ML_Classification


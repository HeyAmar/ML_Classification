import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
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
    return train_test_split(X, y, test_size=0.2, random_state=42)

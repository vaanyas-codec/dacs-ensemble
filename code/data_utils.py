"""
Data loading and preprocessing for the Pima Indians Diabetes and
Heart Disease (Cleveland/UCI) datasets.

Both datasets are expected as local CSV files under data/:
    data/pima_diabetes.csv
    data/heart_disease.csv

Download instructions are in the project README.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_pima(path="data/pima_diabetes.csv"):
    df = pd.read_csv(path)
    # If the file doesn't have headers (raw UCI/Kaggle export), assign them.
    if not set(["Outcome"]).issubset(df.columns):
        df = pd.read_csv(path, names=PIMA_COLUMNS)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    return X, y


def load_heart(path="data/heart_disease.csv"):
    df = pd.read_csv(path)
    if not set(["target"]).issubset(df.columns):
        df = pd.read_csv(path, names=HEART_COLUMNS, na_values="?")
    else:
        df = df.replace("?", np.nan)
    df = df.dropna()
    # Original UCI target is 0 (no disease) to 4 (severity levels).
    # Binarize: 0 = no disease, 1 = disease present (standard practice,
    # matches how most published benchmarks report accuracy).
    df["target"] = (df["target"].astype(float) > 0).astype(int)
    X = df.drop(columns=["target"]).astype(float).values
    y = df["target"].values
    return X, y


def preprocess(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

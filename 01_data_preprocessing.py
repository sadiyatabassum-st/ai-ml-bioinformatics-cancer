"""
01_data_preprocessing.py
--------------------------------------------------------------
Loads the Breast Cancer Wisconsin (Diagnostic) dataset — 569 patient
samples, each with 30 biomarker features computed from digitized
images of fine needle aspirate (FNA) of breast masses (cell nucleus
size, texture, symmetry, etc.). Target: Malignant (M) vs Benign (B).

This mirrors the clinical dataset workflow described in Assignment
Q2/Q3: data loading -> cleaning -> normalization -> feature selection.
--------------------------------------------------------------
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_clean_data():
    """Load dataset, check for missing values, and return a clean DataFrame."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "diagnosis"}, inplace=True)

    # Data cleaning: check for nulls / duplicates (biological datasets are rarely this clean —
    # in a real GEO/TCGA dataset this step would also handle missing probes, batch effects, etc.)
    print(f"Missing values found: {df.isnull().sum().sum()}")
    print(f"Duplicate rows found: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    return df, data.target_names


def split_and_scale(df, test_size=0.2, random_state=42):
    """Split into train/test and standardize features (mean=0, std=1)."""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


if __name__ == "__main__":
    df, target_names = load_and_clean_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts()}")
    print(f"Target classes: {list(target_names)}  (0 = malignant, 1 = benign)")

    df.to_csv(os.path.join(OUT_DIR, "clean_dataset.csv"), index=False)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    print("Saved: data/clean_dataset.csv, data/scaler.pkl")

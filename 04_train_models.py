"""
04_train_models.py
--------------------------------------------------------------
Trains and compares four classification approaches on the biomarker
data — spanning classical ML to deep learning, so the project
demonstrates range (this maps to Assignment Q1's requirement to
cover supervised, and deep-learning-based methods):

  1. Logistic Regression      (interpretable baseline)
  2. Random Forest            (feature-importance friendly, robust)
  3. Support Vector Machine   (strong on high-dimensional biomarker data)
  4. Artificial Neural Network (Keras/TensorFlow — the ANN referenced
                                  in the course material)
--------------------------------------------------------------
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


def build_ann(input_dim):
    """Simple feed-forward ANN for binary classification (malignant vs benign)."""
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_all_models(df):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    trained_models = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)
    results["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test_s))
    trained_models["logistic_regression"] = lr

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_s, y_train)
    results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test_s))
    trained_models["random_forest"] = rf

    # 3. SVM
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_s, y_train)
    results["SVM (RBF)"] = accuracy_score(y_test, svm.predict(X_test_s))
    trained_models["svm"] = svm

    # 4. ANN
    ann = build_ann(X_train_s.shape[1])
    ann.fit(X_train_s, y_train, epochs=60, batch_size=16, verbose=0, validation_split=0.15)
    ann_preds = (ann.predict(X_test_s, verbose=0) > 0.5).astype(int).flatten()
    results["ANN (Keras)"] = accuracy_score(y_test, ann_preds)
    trained_models["ann"] = ann

    return results, trained_models, (X_train_s, X_test_s, y_train, y_test), scaler


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))
    results, trained_models, splits, scaler = train_all_models(df)

    print("Model comparison (test accuracy):")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<22}: {acc:.4f}")

    # Save non-Keras models with joblib, ANN with Keras native format
    joblib.dump(trained_models["logistic_regression"], os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(trained_models["random_forest"], os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(trained_models["svm"], os.path.join(MODEL_DIR, "svm.pkl"))
    trained_models["ann"].save(os.path.join(MODEL_DIR, "ann_model.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved trained models to /models and results.json")

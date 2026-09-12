"""
05_evaluate.py
--------------------------------------------------------------
Full evaluation suite: confusion matrix, ROC-AUC curve, precision/
recall/F1, and a head-to-head model comparison chart. This is the
"evaluation metrics" step called out in Assignment Q3.
--------------------------------------------------------------
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def load_test_split(df):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return scaler.transform(X_test), y_test


def plot_confusion_matrix(model, X_test, y_test, name, keras=False):
    if keras:
        preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    else:
        preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.title(f"Confusion Matrix — {name}", fontsize=12, weight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(IMG_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
    plt.close()

    print(f"\n{name} — classification report:")
    print(classification_report(y_test, preds, target_names=["Malignant", "Benign"]))


def plot_roc_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(7, 6))
    for name, (model, keras) in models_dict.items():
        if keras:
            probs = model.predict(X_test, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison", fontsize=13, weight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "roc_curves.png"), dpi=150)
    plt.close()


def plot_model_comparison(results_path):
    with open(results_path) as f:
        results = json.load(f)

    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    accs = list(results.values())
    bars = plt.bar(names, accs, color=["#3B7DD8", "#3FA34D", "#E8A93D", "#E85D5D"])
    plt.ylim(0.85, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("Model Performance Comparison", fontsize=13, weight="bold")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.003, f"{acc:.3f}",
                  ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))
    X_test, y_test = load_test_split(df)

    lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    svm = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    ann = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ann_model.keras"))

    plot_confusion_matrix(rf, X_test, y_test, "Random Forest")
    plot_confusion_matrix(ann, X_test, y_test, "ANN", keras=True)

    plot_roc_curves({
        "Logistic Regression": (lr, False),
        "Random Forest": (rf, False),
        "SVM": (svm, False),
        "ANN": (ann, True),
    }, X_test, y_test)

    plot_model_comparison(os.path.join(MODEL_DIR, "results.json"))

    print("\nSaved: confusion matrices, roc_curves.png, model_comparison.png")

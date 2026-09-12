"""
02_feature_selection.py
--------------------------------------------------------------
Identifies the biomarkers that matter most for classifying a tumor
as malignant vs benign, using two complementary techniques:

  1. Correlation-based filtering (removes redundant/collinear features)
  2. Recursive Feature Elimination (RFE) with a Random Forest estimator
     to rank features by predictive importance

This step matters a lot in real bioinformatics work: gene expression /
biomarker panels often contain thousands of correlated features, and a
biotech ML pipeline needs to explain *which* biological signals drive
a prediction, not just produce a black-box score.
--------------------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def correlation_heatmap(df, top_n=15):
    """Plot a correlation heatmap for the top N features most correlated with diagnosis."""
    corr = df.corr(numeric_only=True)
    top_features = corr["diagnosis"].abs().sort_values(ascending=False)[1:top_n + 1].index.tolist()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features + ["diagnosis"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", cbar_kws={"label": "Correlation"})
    plt.title("Top 15 Biomarkers Most Correlated with Diagnosis", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    return top_features


def rank_features_rfe(X, y, feature_names, n_features=10):
    """Rank features using Recursive Feature Elimination with a Random Forest."""
    estimator = RandomForestClassifier(n_estimators=200, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X, y)

    ranked = pd.DataFrame({
        "feature": feature_names,
        "selected": selector.support_,
        "rank": selector.ranking_
    }).sort_values("rank")

    plt.figure(figsize=(9, 6))
    top = ranked[ranked["selected"]].sort_values("rank")
    sns.barplot(data=top, y="feature", x=[1] * len(top), color="#3B7DD8")
    plt.title(f"Top {n_features} Selected Biomarkers (RFE + Random Forest)", fontsize=13, weight="bold")
    plt.xlabel("Selected")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_selection.png"), dpi=150)
    plt.close()

    return ranked


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset.csv"))
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    top_corr_features = correlation_heatmap(df)
    print(f"Top correlated features:\n{top_corr_features}\n")

    ranked = rank_features_rfe(X.values, y.values, X.columns.tolist())
    ranked.to_csv(os.path.join(DATA_DIR, "feature_ranking.csv"), index=False)
    print(f"Top selected features:\n{ranked[ranked['selected']]['feature'].tolist()}")
    print("\nSaved: images/correlation_heatmap.png, images/feature_selection.png")

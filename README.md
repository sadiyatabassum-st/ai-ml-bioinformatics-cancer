# AI/ML for Cancer Classification & Biological Sequence Analysis

An end-to-end computational biology project combining **machine learning for cancer classification** with **deep learning for biological DNA sequence analysis**.

The project demonstrates a reproducible workflow covering data preprocessing, exploratory analysis, feature selection, dimensionality reduction, machine-learning model development, hyperparameter optimization, model evaluation, feature interpretation, and biological sequence modeling.

> **Important:** This is an educational and research-oriented computational project. The cancer classification models are **not clinical diagnostic tools** and have not been clinically validated.

---

## Project Overview

This project contains two complementary AI/ML components:

### 1. Cancer Classification

Machine-learning models are trained to classify breast-cancer observations using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The dataset contains numerical morphological measurements derived from digitized images of breast-cell nuclei, rather than gene-expression measurements.

The workflow includes:

* Data preprocessing and quality checks
* Feature scaling
* Exploratory data analysis
* Feature selection/ranking
* PCA analysis
* Multiple machine-learning algorithms
* Hyperparameter optimization
* Cross-validation
* Test-set evaluation
* ROC-AUC analysis
* Confusion-matrix analysis
* Feature-importance interpretation

### 2. Biological DNA Sequence Analysis

A separate deep-learning workflow uses an **LSTM/RNN architecture** to analyze biological DNA sequences and identify sequence patterns.

This demonstrates how deep learning can be applied directly to biological sequence data rather than relying only on manually engineered numerical features.

---

## Project Objectives

The main objectives were to:

1. Build a complete machine-learning workflow for biological data.
2. Compare multiple classical ML algorithms.
3. Apply appropriate preprocessing and feature scaling.
4. Perform feature selection and dimensionality reduction.
5. Optimize model hyperparameters using cross-validation.
6. Evaluate models using metrics beyond accuracy.
7. Investigate biologically meaningful feature patterns.
8. Apply deep learning to DNA sequence data.
9. Combine Python-based ML with R-based visualization.
10. Demonstrate an end-to-end computational R&D workflow.

---

## Project Workflow

```text
Public Biological Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ├──────────────► Feature Ranking
        │
        ▼
Feature Scaling
        │
        ▼
PCA / Dimensionality Analysis
        │
        ▼
Train / Test Split
        │
        ▼
Model Development
        │
        ├── Logistic Regression
        ├── Support Vector Machine
        ├── Random Forest
        └── Artificial Neural Network
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-score
        ├── ROC-AUC
        └── Confusion Matrix
        │
        ▼
Biological Interpretation
```

A separate sequence-learning branch performs:

```text
DNA Sequences
      │
      ▼
Sequence Encoding
      │
      ▼
Embedding / Representation
      │
      ▼
LSTM / RNN
      │
      ▼
Sequence Classification
```

---

## Dataset

### Wisconsin Diagnostic Breast Cancer Dataset

The cancer-classification component uses the publicly available WDBC dataset.

The processed dataset contains:

* **569 observations**
* **30 numerical features**
* **1 diagnosis/classification variable**

The features represent morphological characteristics of cell nuclei, including measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

Both mean, standard-error, and worst-value measurements are represented.

---

## Data Preprocessing

The preprocessing workflow included:

* Dataset inspection
* Data cleaning
* Target encoding
* Feature/target separation
* Train/test splitting
* Feature scaling
* Preparation of model-ready datasets

The workflow was designed to avoid **data leakage**, with preprocessing and model-selection steps handled within the appropriate training workflow.

---

## Machine Learning Models

Four major machine-learning approaches were evaluated:

### Logistic Regression

Used as a strong and interpretable baseline classification model.

### Support Vector Machine

Used to evaluate a margin-based nonlinear/classification approach.

### Random Forest

Used as a tree-based ensemble model capable of capturing nonlinear relationships and providing feature-importance information.

### Artificial Neural Network

A neural-network approach was included to compare classical statistical/ML methods with a simple deep-learning architecture.

---

## Model Comparison

Leakage-aware evaluation produced the following test-set ROC-AUC results:

| Model               | Test ROC-AUC |
| ------------------- | -----------: |
| Logistic Regression |   **0.9954** |
| SVM                 |   **0.9950** |
| Random Forest       |   **0.9937** |
| ANN                 |   **0.9937** |

The results show that all four models performed strongly on this dataset, with Logistic Regression providing the highest test ROC-AUC among the evaluated models.

---

## Hyperparameter Optimization

Grid-search-based optimization with cross-validation was performed to identify suitable model configurations.

Best cross-validation ROC-AUC results:

| Model               | Best CV ROC-AUC |
| ------------------- | --------------: |
| Logistic Regression |      **0.9960** |
| SVM                 |      **0.9894** |
| Random Forest       |      **0.9896** |
| ANN                 |      **0.9897** |

This comparison demonstrates the importance of evaluating model performance using cross-validation rather than relying only on a single test-set metric.

---

## Final Logistic Regression Model

The final Logistic Regression model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **98.25%** |
| Precision | **98.61%** |
| Recall    | **98.61%** |
| F1-score  | **98.61%** |
| ROC-AUC   | **0.9957** |

### Confusion Matrix

```text
                 Predicted
                 Negative  Positive

Actual Negative     41        1
Actual Positive      1       71
```

The model therefore produced only two classification errors on the held-out test set.

---

## Feature Analysis

Feature ranking identified several morphological measurements among the most informative variables.

Examples include:

* **Worst concave points**
* **Mean concave points**
* **Worst area**
* Other radius, perimeter, area, and concavity-related measurements

These findings demonstrate how model-based feature analysis can help identify variables that contribute strongly to classification performance.

Feature ranking results are available in:

```text
data/feature_ranking.csv
```

---

## Biological Sequence Analysis

The second component extends the project from tabular biological measurements to raw biological sequence data.

An LSTM/RNN-based model was developed to learn sequence patterns from DNA sequences.

The sequence-learning component achieved approximately **91.25% test accuracy** on the project sequence-classification task.

The purpose of this component is not clinical prediction. Instead, it demonstrates the application of deep learning to biological sequence-pattern recognition.

This provides a bridge toward applications such as:

* Promoter analysis
* Regulatory sequence analysis
* Motif detection
* Genomic sequence classification
* Computational genomics
* Sequence-based biomarker research

---

## R Analysis

An R-based visualization workflow is included to complement the Python analysis.

The R script demonstrates the use of the R ecosystem for exploratory analysis and visualization.

```text
R/
└── visualization.R
```

---

## Repository Structure

The current GitHub repository is organized according to the files actually uploaded:

```text
ai-ml-bioinformatics-cancer/
│
├── README.md
├── requirements.txt
│
├── 01_data_preprocessing.py
├── 02_feature_selection.py
├── 03_pca_analysis.py
├── 04_train_models.py
├── 05_evaluate.py
├── 06_dna_sequence_rnn.py
│
├── AI_ML_Bioinformatics_Cancer_Sequence_Classification.ipynb
│
├── data/
│   ├── clean_dataset.csv
│   ├── feature_ranking.csv
│   └── scaler.pkl
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── ann_model.keras
│   ├── dna_lstm_model.keras
│   ├── results.json
│   └── scaler.pkl
│
├── R/
│   └── visualization.R
│
└── images/
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## Technologies Used

### Programming

* Python
* R

### Python Libraries

* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn

### Machine Learning

* Logistic Regression
* Support Vector Machine
* Random Forest
* Artificial Neural Network
* LSTM / RNN

### Analysis

* Feature selection
* PCA
* Cross-validation
* Grid search
* ROC-AUC
* Confusion matrices
* Precision, Recall and F1-score

---

## Reproducibility

The repository contains the main analysis scripts, notebook, processed dataset, model artifacts, visualization script, and dependency file required to understand and reproduce the workflow.

The recommended execution order is:

```text
01_data_preprocessing.py
        ↓
02_feature_selection.py
        ↓
03_pca_analysis.py
        ↓
04_train_models.py
        ↓
05_evaluate.py
        ↓
06_dna_sequence_rnn.py
```

The complete workflow can also be reviewed through:

```text
AI_ML_Bioinformatics_Cancer_Sequence_Classification.ipynb
```

---

## R&D Relevance

This project was designed as a **computational R&D portfolio project** rather than simply a machine-learning exercise.

It demonstrates the ability to:

* Translate biological data into computationally analyzable formats
* Perform structured data preprocessing and quality checks
* Compare different modeling approaches
* Apply statistical and machine-learning evaluation methods
* Investigate important biological features
* Work with both tabular biological data and DNA sequences
* Use classical ML alongside deep learning
* Interpret model results rather than relying only on accuracy
* Build a reproducible analytical workflow

These skills are relevant to computational biology, bioinformatics, biomarker research, translational research, scientific data analysis, and AI/ML applications in life sciences.

---

## Final Conclusion

This project demonstrates an end-to-end application of **AI/ML to biological data**, combining two different data modalities: structured cancer-related morphological measurements and biological DNA sequences.

For the cancer-classification component, multiple machine-learning models were systematically compared using leakage-aware preprocessing, cross-validation, hyperparameter optimization, and independent test-set evaluation. Logistic Regression produced the strongest overall performance, achieving a **0.9957 ROC-AUC and 98.25% accuracy** on the held-out test set.

The feature-analysis component further showed that morphological measurements related to **concavity, concave points, area, and radius** contributed strongly to classification.

The DNA sequence component extended the project into deep learning, demonstrating how an LSTM/RNN can learn patterns directly from biological sequences.

### Overall R&D takeaway

The key outcome of this project is not simply the high classification score. The project demonstrates a complete analytical thought process:

**biological data → preprocessing → feature analysis → model development → optimization → rigorous evaluation → biological interpretation**

This provides a foundation for future work using real-world genomic and molecular datasets, including gene-expression data, public cancer cohorts, regulatory DNA sequences, and larger multi-omics datasets.

> **Scope limitation:** The models were developed for research and portfolio purposes using public datasets. The results should not be interpreted as evidence of clinical diagnostic performance or readiness for clinical deployment.

---

## Future Work

Potential extensions include:

* Applying the workflow to real gene-expression datasets from TCGA/GEO
* Multi-class cancer subtype classification
* Integrating genomic and clinical features
* SHAP-based model explainability
* Transformer-based genomic sequence models
* Larger sequence datasets
* External validation on independent datasets
* Model deployment through an API or research application

---

## License

This repository is intended for educational and research purposes.

Please respect the licensing and attribution requirements of the original public datasets and third-party resources used in this project.

---

## Author

**Sadiya Tabassum**

Biotechnology | Bioinformatics | AI/ML | Scientific & Clinical Data Analytics

GitHub: `sadiyatabassum-st`

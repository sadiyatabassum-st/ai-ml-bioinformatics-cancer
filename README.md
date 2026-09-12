# AI/ML for Cancer Classification & Biological Sequence Analysis

## Project Overview

This project demonstrates an end-to-end **AI/ML and bioinformatics workflow** combining cancer-related data analysis with biological sequence classification.

The project contains two complementary components:

1. **Cancer Classification** using machine learning on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
2. **Biological Sequence Analysis** using an LSTM/RNN model to classify DNA sequences containing a promoter-like motif.

The project is designed as an **R&D and portfolio project** to demonstrate the application of machine learning, statistical analysis, biological data processing, and sequence modeling.

> **Important:** This project is intended for research and educational purposes only. The cancer classification model is **not a clinically validated diagnostic tool**.

---

## Project Objectives

* Apply machine learning to biological/cancer-related data.
* Perform data preprocessing and quality checks.
* Compare multiple classification algorithms.
* Apply feature analysis and dimensionality reduction.
* Perform hyperparameter optimization.
* Evaluate models using multiple performance metrics.
* Analyze biologically relevant features.
* Build an LSTM-based model for DNA sequence classification.
* Integrate Python and R-based analysis.
* Demonstrate a reproducible computational biology workflow.

---

# Project Components

## 1. Cancer Classification

The first component uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The dataset contains:

* **569 observations**
* **30 numerical features**
* **Diagnosis:** Benign or Malignant

The features represent **morphological measurements of cell nuclei** derived from digitized breast-cell images.

Examples include:

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

### Machine Learning Models

Four classification algorithms were evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Artificial Neural Network (ANN)

---

# 2. Data Preprocessing

The preprocessing workflow includes:

1. Loading the dataset.
2. Cleaning the data.
3. Separating features and target variables.
4. Encoding the diagnosis labels.
5. Train/test splitting.
6. Feature scaling.
7. Checking data quality.
8. Preparing the dataset for machine learning.

The workflow was designed to avoid data leakage by fitting preprocessing steps only on the training data.

---

# 3. Feature Analysis

Feature analysis was performed to identify variables that contributed strongly to classification performance.

Important features included:

* Worst concave points
* Mean concave points
* Worst area

Feature ranking was generated and stored in:

`data/feature_ranking.csv`

Dimensionality-reduction analysis was also performed using PCA.

---

# 4. Model Development

The following models were trained and compared:

| Model               | Test ROC-AUC |
| ------------------- | -----------: |
| Logistic Regression |   **0.9954** |
| SVM                 |   **0.9950** |
| Random Forest       |   **0.9937** |
| ANN                 |   **0.9937** |

ROC-AUC was used in addition to accuracy to provide a more informative assessment of classification performance.

---

# 5. Hyperparameter Optimization

Grid-search-based hyperparameter optimization was performed for the machine learning models.

| Model               | Best CV ROC-AUC |
| ------------------- | --------------: |
| Logistic Regression |      **0.9960** |
| SVM                 |      **0.9894** |
| Random Forest       |      **0.9896** |
| ANN                 |      **0.9897** |

The optimized Logistic Regression model achieved the strongest cross-validation performance.

---

# 6. Final Logistic Regression Model

The final Logistic Regression model achieved the following test-set performance:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **98.25%** |
| Precision | **98.61%** |
| Recall    | **98.61%** |
| F1 Score  | **98.61%** |
| ROC-AUC   | **0.9957** |

### Confusion Matrix

```text
[[41, 1],
 [ 1, 71]]
```

This indicates:

* 41 correctly classified negative cases
* 71 correctly classified positive cases
* 1 false positive
* 1 false negative

The confusion matrix and ROC curve are available in the `images/` directory.

---

# 7. Biological Sequence Analysis

The second component extends the project from tabular biological data into **DNA sequence modeling**.

An LSTM/RNN-based model was developed to classify biological DNA sequences.

### Sequence characteristics

* DNA sequences were represented using the nucleotide alphabet:

  * A
  * T
  * G
  * C
* The model processes sequence information rather than manually engineered tabular features.
* The task focuses on recognizing a **promoter-like/TATA-box-like motif** within 40 bp DNA sequences.

The LSTM model achieved approximately **91.25% test accuracy** on the sequence-classification task.

This component demonstrates how deep learning can be applied to biological sequence data.

---

# 8. R Analysis

R was used for additional visualization and statistical analysis.

The R script is located at:

```text
R/visualization.R
```

This provides an example of integrating Python-based machine learning workflows with R-based analysis.

---

# 9. Project Workflow

```text
Biological Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Analysis
        │
        ├───────────────┐
        ▼               ▼
Cancer Classification  DNA Sequence Analysis
        │               │
        ▼               ▼
ML Models              LSTM/RNN
        │               │
        ▼               ▼
Hyperparameter         Sequence
Optimization           Classification
        │
        ▼
Model Evaluation
        │
        ▼
Biological Interpretation
```

---

# 10. Repository Structure

```text
ai-ml-bioinformatics-cancer/
│
├── README.md
├── LICENSE
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

# 11. Technologies Used

### Programming

* Python
* R

### Machine Learning

* Scikit-learn
* TensorFlow / Keras
* Logistic Regression
* Support Vector Machine
* Random Forest
* Artificial Neural Network
* LSTM / RNN

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* PCA
* Statistical evaluation

### Bioinformatics

* DNA sequence processing
* Biological sequence classification
* Promoter/motif analysis
* Computational biology workflows

---

# 12. Reproducibility

The repository contains:

* Python analysis scripts
* Jupyter Notebook
* Processed dataset
* Feature-ranking results
* Trained machine-learning models
* LSTM model
* Evaluation results
* R visualization script
* Project dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The individual Python scripts can then be executed according to the workflow.

---

# 13. R&D Relevance

This project demonstrates an end-to-end computational R&D workflow:

```text
Biological Data
      ↓
Data QC & Preprocessing
      ↓
Feature Analysis
      ↓
Model Development
      ↓
Model Optimization
      ↓
Rigorous Evaluation
      ↓
Biological Interpretation
```

The project is particularly relevant to areas such as:

* Bioinformatics
* Computational Biology
* AI/ML in Life Sciences
* Biomarker research
* Biological data analysis
* Genomic sequence analysis
* Scientific data science
* Translational research

---

# 14. Final Conclusion

This project demonstrates the application of **AI/ML methods across two different types of biological data**: structured cancer-related measurements and raw DNA sequences.

The cancer-classification component established a complete machine-learning workflow involving preprocessing, feature analysis, model comparison, hyperparameter optimization, and rigorous evaluation. Among the evaluated models, Logistic Regression achieved the strongest overall performance, with a test ROC-AUC of **0.9957** and an accuracy of **98.25%**.

The biological sequence component demonstrated how an **LSTM/RNN architecture** can be applied directly to DNA sequences to identify promoter-like sequence patterns, achieving approximately **91.25% test accuracy**.

Overall, the project demonstrates the complete progression:

**biological data → preprocessing → feature analysis → model development → optimization → rigorous evaluation → biological interpretation**

It provides a foundation that can be extended to larger and more complex biological datasets and demonstrates practical skills relevant to **bioinformatics, computational biology, AI/ML, and scientific R&D**.

> The results presented in this repository are for research and educational purposes and should not be interpreted as clinical diagnostic performance.

---

# 15. Future Work

Potential future extensions include:

* Applying the workflow to real gene-expression datasets such as TCGA or GEO.
* Multi-class cancer subtype classification.
* Explainable AI using SHAP or LIME.
* Larger biological sequence datasets.
* Transformer-based genomic models such as DNABERT.
* External validation using independent datasets.
* Integration of additional omics data.
* Development of an interactive research application or API.
* Benchmarking classical ML models against deep-learning approaches.

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

The MIT License applies to the original code in this repository. Dataset ownership and third-party software/library licenses remain subject to their respective terms.

---

# Author

**Sadiya Tabassum**

Biotechnology | Bioinformatics | AI/ML | Scientific & Clinical Data Analytics

* GitHub: `sadiyatabassum-st`
* LinkedIn: https://www.linkedin.com/in/sadiyatabassum8015/

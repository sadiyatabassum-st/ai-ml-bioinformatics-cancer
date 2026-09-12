# AI/ML for Cancer Classification & Biological Sequence Analysis

## Project Overview

This project demonstrates the application of **machine learning and deep learning techniques in bioinformatics** for two complementary biological data-analysis tasks:

1. **Cancer classification** using gene-expression-derived clinical/biomarker features.
2. **Biological sequence analysis** using a Long Short-Term Memory (LSTM) neural network.

The project was developed as an **AI/ML in Bioinformatics portfolio project** with an emphasis on reproducible data preprocessing, model comparison, feature analysis, performance evaluation, and biological interpretation.

---

## Objectives

* Perform quality-controlled preprocessing of biological/clinical data.
* Build and compare multiple machine-learning classification models.
* Identify informative features associated with cancer classification.
* Evaluate models using appropriate classification metrics.
* Apply deep learning to biological sequence data.
* Demonstrate how AI/ML approaches can support computational biology and R&D workflows.

---

## Project Workflow

```text
Biological / Clinical Data
          │
          ▼
    Data Quality Check
          │
          ▼
   Data Preprocessing
          │
          ▼
 Feature Selection / Ranking
          │
          ▼
 ┌───────────────────────────┐
 │   Machine Learning Models │
 ├───────────────────────────┤
 │ Logistic Regression       │
 │ Support Vector Machine    │
 │ Random Forest             │
 │ Artificial Neural Network │
 └───────────────────────────┘
          │
          ▼
 Model Evaluation & Comparison
          │
          ▼
 Feature Importance
          │
          ▼
 Cancer Classification

              +

 Biological Sequence Data
          │
          ▼
 Sequence Preprocessing
          │
          ▼
       LSTM / RNN
          │
          ▼
 Sequence Classification /
 Pattern Learning
```

---

## Dataset

The cancer classification component uses the **Wisconsin Diagnostic Breast Cancer dataset**, containing:

* **569 observations**
* **30 numerical features**
* **1 target variable (****`diagnosis`****)**

The features describe characteristics calculated from digitized images of breast-cell nuclei.

The data were divided into training and testing sets:

* Training set: **455 samples**
* Test set: **114 samples**

The target distribution in the training set was:

* 285 samples
* 170 samples

The test set contained:

* 72 samples
* 42 samples

---

## Data Preprocessing

The preprocessing workflow includes:

* Dataset inspection
* Data cleaning
* Separation of predictors and target
* Encoding of the target variable
* Train/test splitting
* Feature scaling
* Feature analysis
* Prevention of data leakage by fitting preprocessing steps using training data

The preprocessing pipeline was designed so that information from the test set was not used during model training.

---

## Machine Learning Models

Four classification approaches were evaluated:

### 1. Logistic Regression

Used as a strong and interpretable baseline classification model.

### 2. Support Vector Machine

Used to evaluate a margin-based classification approach for the cancer dataset.

### 3. Random Forest

An ensemble tree-based model used to capture nonlinear relationships and provide feature-importance information.

### 4. Artificial Neural Network

A feed-forward neural-network approach was evaluated alongside the traditional machine-learning models.

---

## Model Comparison

The models were evaluated using metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* ROC curve

The leakage-safe model comparison produced the following test ROC-AUC results:

| Model                     | Test ROC-AUC |
| ------------------------- | -----------: |
| Logistic Regression       |   **0.9954** |
| Support Vector Machine    |   **0.9950** |
| Random Forest             |   **0.9937** |
| Artificial Neural Network |   **0.9937** |

These results demonstrate strong classification performance across all four approaches.

---

## Hyperparameter Optimization

Hyperparameter tuning was performed using cross-validation.

The best cross-validation ROC-AUC results included:

| Model                     | Best CV ROC-AUC |
| ------------------------- | --------------: |
| Logistic Regression       |      **0.9960** |
| Support Vector Machine    |      **0.9894** |
| Random Forest             |      **0.9896** |
| Artificial Neural Network |      **0.9897** |

For the final model, Logistic Regression provided the strongest combination of performance and interpretability.

---

## Final Logistic Regression Model

The final Logistic Regression model achieved:

* **Accuracy:** 98.25%
* **Precision:** 98.61%
* **Recall:** 98.61%
* **F1-score:** 98.61%
* **ROC-AUC:** 0.9957

Confusion matrix:

```text
[[41, 1],
 [ 1, 71]]
```

This indicates that only two observations were misclassified in the held-out test set.

---

## Feature Analysis

Feature analysis was performed to investigate which biological measurements contributed most strongly to model predictions.

Examples of highly ranked features include:

* Worst concave points
* Mean concave points
* Worst area

Feature-ranking outputs are included in the project files for further analysis.

---

## Biological Sequence Analysis

In addition to cancer classification, the project explores **biological sequence analysis using deep learning**.

An LSTM-based neural-network approach was implemented to demonstrate how recurrent neural networks can learn patterns from biological sequences.

The sequence-analysis component provides a separate computational-biology workflow from the tabular cancer-classification task.

The implementation is included in:

```text
src/06_dna_sequence_rnn.py
```

and the trained sequence model is stored under the project model files.

---

## R Analysis

An R script is included for additional visualization and analysis:

```text
R/visualization.R
```

This demonstrates the use of both **Python and R** within a computational biology workflow.

---

## Repository Structure

```text
AI-ML-Bioinformatics/
│
├── README.md
│
├── notebooks/
│   └── AI_ML_Bioinformatics_Cancer_Sequence_Classification.ipynb
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
│   ├── scaler.pkl
│   └── results.json
│
├── src/
│   ├── 02_feature_selection.py
│   ├── 04_model_training.py
│   ├── 06_dna_sequence_rnn.py
│   └── ...
│
├── R/
│   └── visualization.R
│
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── ...
│
└── requirements.txt
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
* Hyperparameter optimization
* Cross-validation
* ROC-AUC analysis

### Deep Learning

* Artificial Neural Networks
* LSTM / RNN
* Biological sequence analysis

### Bioinformatics / Computational Biology

* Biological sequence processing
* Feature analysis
* Cancer biomarker-related classification
* Computational analysis of biological data

---

## Key Outcomes

This project demonstrates the ability to:

* Build an end-to-end machine-learning workflow.
* Perform data preprocessing while avoiding test-set leakage.
* Compare multiple ML algorithms systematically.
* Perform hyperparameter optimization.
* Evaluate classification models using clinically relevant performance metrics.
* Analyze feature importance.
* Apply deep learning to biological sequence data.
* Combine computational biology with machine learning.
* Organize an analysis into reproducible Python scripts and notebooks.

---

## R&D Relevance

The project is designed as a demonstration of how AI/ML can be incorporated into **biological and pharmaceutical R&D workflows**.

Potential applications of similar approaches include:

* Biomarker analysis
* Molecular classification
* Biological sequence analysis
* Pattern recognition in experimental datasets
* Predictive modeling
* Exploratory analysis of high-dimensional biological data
* Supporting data-driven research decisions

This project is intended as a **computational research and portfolio demonstration**, not as a clinically validated diagnostic system.

---

## Reproducibility

The main end-to-end workflow is available in:

```text
notebooks/
AI_ML_Bioinformatics_Cancer_Sequence_Classification.ipynb
```

Supporting Python scripts, models, data-processing outputs, R analysis, and visualization files are organized into separate directories.

---

## Author

**Sadiya Tabassum**

Biotechnology | Bioinformatics | AI/ML | Scientific & Clinical Data Analytics

GitHub: `sadiyatabassum-st`

---

## License

This project is intended for educational, research, and portfolio purposes.

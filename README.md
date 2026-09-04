# Operation Risk Handling Index (ORHI)

A semi-supervised Machine Learning system to classify **human operational risk** into Low, Medium, and High categories — built on behavioral survey data and deployed as a real-time web application.

🔗 **Live App:** [operation-risk-handling-predictor.streamlit.app](https://operation-risk-handling-predictor.streamlit.app/)

---

## Problem Statement

Organizations face significant losses due to human operational risk — errors, misjudgments, and behavioral patterns that lead to failures in critical processes. This project builds a data-driven pipeline to classify individuals into risk tiers based on behavioral features, enabling proactive risk management.

---

## Pipeline Overview

```
Survey Data (200+ responses)
        ↓
Preprocessing & Feature Engineering
        ↓
GMM Clustering (Unsupervised Label Generation)
        ↓
Supervised Classification (5 Models)
        ↓
SHAP Analysis (Explainability)
        ↓
Streamlit Deployment (Real-time Predictions)
```

---

## Key Features

- **Semi-supervised approach:** No labeled data available initially — used GMM clustering with BIC optimization to generate risk pseudo-labels (Low / Medium / High)
- **5 Classifiers benchmarked:** Logistic Regression, Decision Tree, SVM, KNN, Random Forest
- **5-Fold Cross Validation** for robust model evaluation
- **Best results:** SVM — 76.2% accuracy | Random Forest — 0.752 F1 / 0.89 AUC
- **SHAP analysis** to identify top behavioral risk predictors
- **Streamlit app** with personalized risk recommendations per user

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Library | Scikit-learn |
| Clustering | GMM (Gaussian Mixture Model) |
| Explainability | SHAP |
| Web App | Streamlit |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Dataset

- **Source:** Primary survey data (Google Forms)
- **Size:** 200+ responses
- **Features:** 8 behavioral features (work habits, decision-making patterns, stress response, etc.)
- **Labels:** Generated via GMM clustering (no manual labeling)

---

## Model Performance

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| SVM | **76.2%** | 0.741 | 0.87 |
| Random Forest | 74.5% | **0.752** | **0.89** |
| Logistic Regression | 71.3% | 0.708 | 0.84 |
| KNN | 69.8% | 0.694 | 0.82 |
| Decision Tree | 68.4% | 0.676 | 0.79 |

*All models evaluated using 5-Fold Cross Validation*

---

## SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to identify which behavioral features most influence risk classification — making the model interpretable and explainable for real-world use.

---

## Run Locally

```bash
git clone https://github.com/Sauve9119/Operation-risk-handling-index.git
cd Operation-risk-handling-index
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
├── OperationRiskHandlingIndex.ipynb  # Full ML pipeline
├── app.py                            # Streamlit web app
├── responses.csv                     # Survey dataset
├── requirements.txt                  # Dependencies
└── README.md
```

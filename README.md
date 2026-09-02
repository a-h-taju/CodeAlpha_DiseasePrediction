# CodeAlpha_DiseasePrediction

** Disease Prediction from Medical Data** | CodeAlpha Machine Learning Internship

A machine learning model that predicts whether a tumour is malignant (cancer) or
benign, based on 30 measurements taken from a patient's biopsy image.

---

## Problem Statement

Early cancer detection saves lives. This project treats screening as a **binary
classification** problem and compares four algorithms to find which one separates
malignant from benign cases most reliably.

An important distinction in medical machine learning: **a missed disease case is far
worse than a false alarm.** A false alarm leads to one extra test; a missed case can
be fatal. That is why recall on the disease class matters more here than raw accuracy.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — built into scikit-learn, no download needed.

| | |
|---|---|
| Patients | 569 |
| Features | 30 medical measurements |
| Missing values | 0 |
| Class balance | 357 benign / 212 malignant |
| Target | 1 = benign (healthy), 0 = malignant (disease) |

Features include radius, texture, perimeter, area, smoothness, compactness and
concavity — each recorded as a mean, a standard error, and a worst-case value.

---

## Approach

1. **Loading** — dataset pulled directly from scikit-learn, verified for missing values.
2. **Splitting** — 80% train / 20% test, stratified so both sets keep the same disease ratio.
3. **Scaling** — medical features have wildly different ranges (area is in the thousands,
   smoothness is around 0.1). Logistic Regression and SVM are wrapped in a `Pipeline`
   with `StandardScaler` so no single feature dominates purely because of its units.
   Tree-based models skip scaling, since they don't need it.
4. **Modelling** — four algorithms compared: Logistic Regression, SVM (RBF kernel),
   Random Forest, and XGBoost.
5. **Validation** — 5-fold stratified cross-validation on the training set, so the
   reported score reflects stable performance rather than one lucky split.
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score and ROC-AUC on the held-out test set.

---

## Results

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.9959** | **0.9825** | 0.9861 | 0.9861 | **0.9861** | **0.9954** |
| SVM (RBF) | 0.9956 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| XGBoost | 0.9941 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9940 |
| Random Forest | 0.9906 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9934 |

**Best model: Logistic Regression (ROC-AUC 0.9954)**

Cross-validation standard deviation was only ±0.005, confirming the score is stable
and not the result of a favourable split.

### Per-class performance (best model)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Disease (malignant) | 0.9762 | 0.9762 | 0.9762 | 42 |
| Healthy (benign) | 0.9861 | 0.9861 | 0.9861 | 72 |

Out of 42 malignant cases in the test set, the model correctly identified 41 and
missed 1. It also raised 1 false alarm on a healthy patient. In a screening context
that trade-off is acceptable — the false alarm costs one follow-up test, while the
single missed case is the error worth reducing further.

The top 10 most influential medical features are ranked in
`task4_disease_prediction_results.png`.

### Notable finding

The simplest model won. Logistic Regression, a linear model, beat both Random Forest
and XGBoost — which suggests the malignant and benign groups are close to **linearly
separable** in this feature space. Complex models are not automatically better, and a
linear model has a real advantage in healthcare: its coefficients can be explained to
a doctor, while an ensemble of 400 trees cannot.

---

## Files

```
task4_disease_prediction.py            # complete, commented source code
task4_disease_prediction_results.png   # confusion matrix, ROC curves, feature importance
requirements.txt
README.md
```

## Setup and Run

```bash
pip install -r requirements.txt
python task4_disease_prediction.py
```

Runs in under a minute on CPU. No internet connection or dataset download required.

---

## Key Takeaways

- Metric choice depends on the domain — in healthcare, **recall on the disease class**
  outranks overall accuracy.
- Cross-validation with a low standard deviation is what makes a high score trustworthy;
  a single 98% result on one split proves much less.
- Model complexity should be justified by results, not assumed. Here the linear model
  won on both performance and interpretability.

### Possible improvements
- Threshold tuning to push disease recall toward 100%, accepting more false alarms
- SHAP values to show which measurements drove each individual prediction
- Validation on an external hospital dataset to test real-world generalisation

---

## Disclaimer

This is an educational project. It is screening-support research code and must not be
used for actual medical diagnosis.

---

## Author

Machine Learning Intern @ CodeAlpha

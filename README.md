# Logistic Regression from Scratch — Customer Churn Prediction

A from-scratch implementation of logistic regression (NumPy only — no scikit-learn in the model itself), 
validated against scikit-learn's implementation, applied to real-world customer churn data.

Built to demonstrate: the underlying math (MLE, cross-entropy, gradient derivation), a correct 
numerically-stable implementation, and proper evaluation practice on an imbalanced classification problem.

---

## 1. Problem

[TO FILL IN STAGE 5 — one paragraph: what is churn, why predict it, what the dataset contains, 
class balance, n rows/features]

---

## 2. Math Summary

### 2.1 The Model
The model estimates the probability of the positive class as:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

### 2.2 Cost Function (Binary Cross-Entropy, derived from Maximum Likelihood)

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

### 2.3 Gradients

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}) \qquad
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

[TO FILL IN STAGE 2 — link to full derivation, either inline or in `notebooks/derivations.ipynb`]

---

## 3. Project Structure

logistic-regression-scratch/
├── src/
│   ├── model.py          # LogisticRegressionScratch class
│   └── train.py          # training script, entry point
├── notebooks/             # exploration, derivations
├── tests/
│   └── test_model.py      # unit tests
├── data/                   # raw / processed dataset (gitignored if large)
├── outputs/
│   └── figures/            # cost curve, comparison plots
├── requirements.txt
└── README.md
---

## 4. How to Run

```bash
git clone https://github.com/Percy5598/logistic-regression-scratch.git
cd logistic-regression-scratch
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

---

## 5. Results

[TO FILL IN STAGE 5/6 — after fitting]

| Metric | From-Scratch Model | scikit-learn |
|---|---|---|
| Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| F1 | — | — |
| ROC-AUC | — | — |

**Cost curve:**
![cost curve](outputs/figures/cost_curve.png)

[TO FILL IN — brief note on any discrepancy vs sklearn, e.g. regularization defaults]

---

## 6. Key Design Decisions

- Numerically stable sigmoid (avoids overflow for large negative `z`)
- Gradient descent implemented as [batch/mini-batch/stochastic — TBD Stage 3]
- Optional L1/L2 regularization (Stage 6)

---

## 7. What I'd Do Differently / Next Steps

[Optional but strong for a portfolio — e.g., feature engineering, cross-validation, handling class imbalance beyond metrics]
---

## 4. How to Run

```bash
git clone https://github.com/<your-username>/logistic-regression-scratch.git
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
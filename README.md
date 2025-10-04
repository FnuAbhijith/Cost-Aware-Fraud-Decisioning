

# Cost-Aware Fraud Decisioning  
**XGBoost + scikit-learn pipeline · Cost-optimal threshold · FastAPI service · SHAP explainability**

A production-style fraud scoring system that **minimizes expected loss** (not just maximizes AUC).  
Includes a ready-to-run **FastAPI** endpoint, training pipeline, cost/PR curve reports, and SHAP explanations.

> **Why this matters:** Real fraud systems choose a **decision threshold** that balances the cost of **missed fraud (FN)** vs **false alarms (FP)**. This project picks that threshold by **minimizing expected business cost**.

---

## 🔎 Problem Statement
Given transaction features (amount, device risk, geo distance, etc.), predict fraud probability and convert it into a **decision** that minimizes:
\[
\textbf{Expected Cost}(t) = C_{FN}\cdot FN(t) + C_{FP}\cdot FP(t)
\]

---

## 🎯 Goals
- Train an XGBoost classifier with a clean scikit-learn preprocessing pipeline.
- Pick a **cost-optimal threshold** (by scanning unique predicted scores).
- Serve real-time scoring via **FastAPI** (`/score`, `/health`) with an API key.
- Provide **explainability** (global SHAP summary; local top-k ready to enable).

---

## 📊 Demo Results (synthetic data included)
> Your numbers will differ if you retrain or swap data.

- **PR-AUC:** `0.8469`  
- **ROC-AUC:** `0.7021`  
- **Cost-optimal threshold:** `0.0087` (with `C_FN=20`, `C_FP=1`)


## 🛠️ Tech Stack
**Python**, **pandas**, **numpy**, **scikit-learn**, **XGBoost**, **SHAP**, **matplotlib**, **FastAPI**, **Uvicorn**

---

## ⚙️ Setup
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt

# Windows
set API_KEY=change-me
uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1

# macOS/Linux
# export API_KEY=change-me
# uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1



---

## 🧱 Architecture

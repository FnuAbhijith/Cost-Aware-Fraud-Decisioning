import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap

DATA = "data/transactions.csv"
ART_DIR = "model/artifacts"
os.makedirs(ART_DIR, exist_ok=True)
COST_FN, COST_FP = 20.0, 1.0

df = pd.read_csv(DATA)
y = df["is_fraud"].astype(int)
X = df.drop(columns=["is_fraud"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

pre = ColumnTransformer([
    ("num", "passthrough", num_cols),
   ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

xgb = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="aucpr", n_jobs=-1, random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", xgb)]).fit(X_tr, y_tr)
proba = pipe.predict_proba(X_te)[:, 1]
roc = roc_auc_score(y_te, proba)
ap  = average_precision_score(y_te, proba)

# cost-optimal threshold
order = np.argsort(proba)[::-1]
y_sorted = y_te.values[order]
proba_sorted = proba[order]
tp = np.cumsum(y_sorted)
fp = np.cumsum(1 - y_sorted)
pos = y_te.sum(); neg = len(y_te) - pos
unique_idx = np.where(np.r_[True, proba_sorted[1:] != proba_sorted[:-1]])[0]
tp_u = tp[unique_idx]; fp_u = fp[unique_idx]
fn_u = pos - tp_u
costs = COST_FN * fn_u + COST_FP * fp_u
best_i = int(costs.argmin())
best_thr = float(proba_sorted[unique_idx][best_i])

print(f"ROC AUC: {roc:.4f} | PR-AUC: {ap:.4f} | Best threshold: {best_thr:.4f}")

# SHAP setup
X_bg = X_tr.sample(min(200, len(X_tr)), random_state=42)
ohe = pre.named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
final_cols = num_cols + cat_names
explainer = shap.TreeExplainer(xgb)

joblib.dump(pipe, f"{ART_DIR}/pipeline.pkl")
joblib.dump({"threshold": best_thr, "roc_auc": roc, "ap": ap}, f"{ART_DIR}/metrics.pkl")
joblib.dump({"explainer": explainer, "final_cols": final_cols}, f"{ART_DIR}/shap.pkl")
print("Saved artifacts →", ART_DIR)

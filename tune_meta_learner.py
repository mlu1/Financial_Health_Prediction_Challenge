"""
Tune meta-learner for stacking using OOF meta-features produced from base models.
Saves best meta-learner and tuning results.

Key fixes vs previous version:
- OOF meta-features are generated using fold-fitted preprocessing (NO leakage).
- Meta-learner is tuned using weighted F1 (not accuracy).
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from data_loader import load_train_data, load_test_data
from preprocessing import PreprocessingPipeline
import config

import os
import warnings

# Python warnings
warnings.filterwarnings("ignore")

# Optional: reduce noisy backend logs
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # if TensorFlow is installed somewhere
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # if HF tokenizers are installed


# --------------------------
# Scoring
# --------------------------
f1w = make_scorer(f1_score, average="weighted")


# --------------------------
# Load data
# --------------------------
print("\nMeta-learner tuning: building leak-free OOF meta-features...")

X_raw, y_raw = load_train_data()

# Encode target variable
le = LabelEncoder()
ordered_classes = ["Low", "High", "Medium"]  # keep your explicit order
le.fit(ordered_classes)

y_encoded = le.transform(y_raw)
classes_encoded = np.unique(y_encoded)
n_classes = len(classes_encoded)
print(f"Target classes encoded: {dict(zip(le.classes_, classes_encoded))}")

# --------------------------
# Base models
# --------------------------
base_models = [
    ("rf", RandomForestClassifier(
        n_estimators=400, max_depth=14, random_state=config.RANDOM_SEED, n_jobs=-1,
        class_weight="balanced"
    )),
    ("gb", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_SEED
    )),
    ("xgb", XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=config.RANDOM_SEED,
        eval_metric="mlogloss"
    )),
    ("cat", CatBoostClassifier(
        n_estimators=500, learning_rate=0.05, depth=5,
        random_state=config.RANDOM_SEED,
        verbose=0
    )),
    ("lgbm", lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )),
]

# --------------------------
# OOF meta-feature generation (NO LEAKAGE)
# --------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

base_model_names = [name for name, _ in base_models]
meta_feature_columns = [f"{name}_c{c}" for name in base_model_names for c in classes_encoded]
meta_feature_columns.append("cluster")  # optional extra feature if present after preprocessing

meta_features = np.zeros((len(X_raw), len(meta_feature_columns)), dtype=float)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_encoded), 1):
    print(f"Fold {fold}/5")

    X_tr_raw = X_raw.iloc[train_idx].copy()
    X_va_raw = X_raw.iloc[val_idx].copy()
    y_tr = y_encoded[train_idx]
    y_va = y_encoded[val_idx]

    # Fit preprocessing ONLY on training fold
    pp = PreprocessingPipeline()
    X_tr = pp.fit_transform(X_tr_raw, pd.Series(le.inverse_transform(y_tr), index=X_tr_raw.index))
    X_va = pp.transform(X_va_raw)

    # Train base models on processed fold data
    for m, (name, model) in enumerate(base_models):
        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)
        probs = model_fold.predict_proba(X_va)

        for c, cls in enumerate(classes_encoded):
            meta_features[val_idx, m * n_classes + c] = probs[:, c]

    # add cluster if available
    if "cluster" in X_va.columns:
        meta_features[val_idx, -1] = X_va["cluster"].to_numpy()
    else:
        meta_features[val_idx, -1] = 0.0

# Save meta-features for inspection
os.makedirs(os.path.join(config.BASE_DIR, "analysis"), exist_ok=True)
meta_df = pd.DataFrame(meta_features, columns=meta_feature_columns)
meta_df["Target"] = y_encoded
meta_df.to_csv(os.path.join(config.BASE_DIR, "analysis", "meta_features_oof.csv"), index=False)
print("Saved OOF meta-features to analysis/meta_features_oof.csv")


# --------------------------
# Meta-learner wrapper (to grid over different estimator classes)
# --------------------------
class EstimatorWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator if estimator is not None else LogisticRegression()

    def fit(self, X, y):
        self.clf_ = clone(self.estimator)
        self.clf_.fit(X, y)
        if hasattr(self.clf_, "classes_"):
            self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.clf_, "predict_proba"):
            return self.clf_.predict_proba(X)
        # fallback: convert decision_function -> pseudo-proba (rare)
        df = self.clf_.decision_function(X)
        e = np.exp(df - df.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# --------------------------
# Tune meta-learner (WEIGHTED F1)
# --------------------------
X_meta = pd.DataFrame(meta_features, columns=meta_feature_columns)
y_meta = y_encoded

grid_params = []

# Logistic Regression (strong baseline meta)
grid_params.append({
    "estimator": [LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced"
    )],
    "estimator__C": [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
})

'''
# LightGBM meta
grid_params.append({
    "estimator": [lgb.LGBMClassifier(
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )],
    "estimator__n_estimators": [200, 400, 800],
    "estimator__learning_rate": [0.01, 0.03, 0.05],
    "estimator__num_leaves": [15, 31, 63],
    "estimator__subsample": [0.8, 0.9, 1.0],
    "estimator__colsample_bytree": [0.8, 0.9, 1.0],
})
'''

# RandomForest meta
grid_params.append({
    "estimator": [RandomForestClassifier(
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )],
    "estimator__n_estimators": [200, 500],
    "estimator__max_depth": [6, 10, None],
    "estimator__min_samples_leaf": [1, 3, 5],
})

# GradientBoosting meta
grid_params.append({
    "estimator": [GradientBoostingClassifier(random_state=config.RANDOM_SEED)],
    "estimator__n_estimators": [150, 300],
    "estimator__learning_rate": [0.03, 0.05, 0.1],
    "estimator__max_depth": [3, 4],
})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
wrapper = EstimatorWrapper()

gs = GridSearchCV(
    wrapper,
    grid_params,
    cv=cv,
    scoring=f1w,          # <-- FIX: weighted F1
    n_jobs=-1,
    verbose=2,
)

print("Starting GridSearchCV for meta-learner (weighted F1)...")
gs.fit(X_meta, y_meta)

print("Best weighted-F1:", gs.best_score_)
print("Best params:", gs.best_params_)

best_est = gs.best_estimator_.clf_

os.makedirs(config.MODEL_DIR, exist_ok=True)
with open(os.path.join(config.MODEL_DIR, "meta_learner_tuned.pkl"), "wb") as f:
    pickle.dump(best_est, f)
print("Saved tuned meta-learner to models/meta_learner_tuned.pkl")

results_df = pd.DataFrame(gs.cv_results_)
results_df.to_csv(os.path.join(config.BASE_DIR, "analysis", "meta_learner_tuning_results.csv"), index=False)
print("Saved tuning results to analysis/meta_learner_tuning_results.csv")


# --------------------------
# Fit FINAL preprocessing on FULL train and save it (for inference)
# --------------------------
print("Fitting final preprocessing pipeline on full training data...")
final_pp = PreprocessingPipeline()
X_full_proc = final_pp.fit_transform(X_raw, y_raw)

with open(config.PREPROCESSING_FILE, "wb") as f:
    pickle.dump(final_pp, f)
print(f"Saved preprocessing pipeline to {config.PREPROCESSING_FILE}")


# --------------------------
# Generate stacked submission
# --------------------------
print("Generating stacked submission with tuned meta-learner...")

X_test_raw, test_ids = load_test_data()
X_test_proc = final_pp.transform(X_test_raw)

# base-model probs on FULL train processed data
test_meta = np.zeros((len(X_test_proc), len(meta_feature_columns)), dtype=float)
for m, (name, model) in enumerate(base_models):
    model_full = clone(model)
    model_full.fit(X_full_proc, y_encoded)
    probs = model_full.predict_proba(X_test_proc)
    for c, cls in enumerate(classes_encoded):
        test_meta[:, m * n_classes + c] = probs[:, c]

if "cluster" in X_test_proc.columns:
    test_meta[:, -1] = X_test_proc["cluster"].to_numpy()
else:
    test_meta[:, -1] = 0.0

test_meta_df = pd.DataFrame(test_meta, columns=meta_feature_columns)

tuned_preds = best_est.predict(test_meta_df)
tuned_preds_decoded = le.inverse_transform(tuned_preds)

os.makedirs(os.path.join(config.BASE_DIR, "submissions"), exist_ok=True)
submission = pd.DataFrame({"ID": test_ids, "Target": tuned_preds_decoded})
submission_path = os.path.join(config.BASE_DIR, "submissions", "submission_stacked_tuned.csv")
submission.to_csv(submission_path, index=False)

print(f"Saved tuned stacked submission to {submission_path}")
print("Top 5 rows:\n", submission.head())
print("Class distribution:\n", submission["Target"].value_counts(normalize=True).rename("proportion"))

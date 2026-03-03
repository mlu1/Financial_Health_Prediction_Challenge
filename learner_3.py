# baseline_catboost_f1.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def feature_engineering(train, test):
    # normalize any case-variant of the target column to 'target'
    target_col = None
    for c in train.columns:
        if c.lower() == "target":
            target_col = c
            break
    if target_col and target_col != "target":
        train = train.rename(columns={target_col: "target"})

    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = -1

    df = pd.concat([train, test], axis=0)

    # -----------------------
    # Missing indicators
    # -----------------------
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col + "_missing"] = df[col].isnull().astype(int)

    # -----------------------
    # Frequency Encoding
    # -----------------------
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "target" in cat_cols:
        cat_cols.remove("target")

    for col in cat_cols:
        freq = df[col].value_counts()
        df[col + "_freq"] = df[col].map(freq)

    # -----------------------
    # Count Features
    # -----------------------
    for col in ["county", "ward", "trainer"]:
        if col in df.columns:
            df[col + "_count"] = df.groupby(col)[col].transform("count")

    # -----------------------
    # Interaction Features
    # -----------------------
    if "county" in df.columns and "belong_to_cooperative" in df.columns:
        df["county_coop"] = (
            df["county"].astype(str) + "_" + df["belong_to_cooperative"].astype(str)
        )

    # -----------------------
    # Date Features
    # -----------------------
    if "first_training_date" in df.columns:
        df["first_training_date"] = pd.to_datetime(
            df["first_training_date"], errors="coerce"
        )
        df["train_year"] = df["first_training_date"].dt.year
        df["train_month"] = df["first_training_date"].dt.month

    # -----------------------
    # Restore train/test
    # -----------------------
    train = df[df["is_train"] == 1].drop(["is_train"], axis=1)
    test = df[df["is_train"] == 0].drop(["is_train", "target"], axis=1)

    return train, test


def target_encode_fold_safe(train, test, cols, target, n_splits=5, seed=42, alpha=10):
    """Fold-safe target encoding for multiclass targets.

    Produces for each categorical column `col` and each class `k` a feature
    `col_te_cls{k}` representing the smoothed fraction of class `k` within
    category `col`. Encodings for training rows are computed out-of-fold to
    avoid leakage; encodings for `test` are computed using full-train stats.
    """
    train = train.copy()
    test = test.copy()

    classes = np.sort(train[target].unique())

    # prepare new columns
    for col in cols:
        for k in classes:
            train[f"{col}_te_cls{k}"] = np.nan
            test[f"{col}_te_cls{k}"] = np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in skf.split(train, train[target]):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]

        for col in cols:
            # counts per category and per class in this fold's train
            grp = tr.groupby(col)[target].value_counts().unstack(fill_value=0)
            counts = tr.groupby(col).size()

            for k in classes:
                prior = (tr[target] == k).mean()
                if k in grp.columns:
                    count_k = grp[k]
                else:
                    count_k = 0
                # smoothing
                mean_k = (count_k + prior * alpha) / (counts + alpha)

                # map to validation rows
                mapped = val[col].map(mean_k)
                mapped = mapped.fillna(prior)
                train.loc[val.index, f"{col}_te_cls{k}"] = mapped

    # compute full-train stats for test
    full = train.copy()
    for col in cols:
        grp = full.groupby(col)[target].value_counts().unstack(fill_value=0)
        counts = full.groupby(col).size()
        for k in classes:
            prior = (full[target] == k).mean()
            if k in grp.columns:
                count_k = grp[k]
            else:
                count_k = 0
            mean_k = (count_k + prior * alpha) / (counts + alpha)
            mapped = test[col].map(mean_k)
            mapped = mapped.fillna(prior)
            test[f"{col}_te_cls{k}"] = mapped

    return train, test


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

TARGET = "target"
ID_COL = "ID"

train, test = feature_engineering(train, test)

# Encode target
le = LabelEncoder()
train[TARGET] = le.fit_transform(train[TARGET])
train, test = target_encode_fold_safe(
    train,
    test,
    cols=[c for c in train.select_dtypes(include=["object"]).columns.tolist() if c not in (ID_COL, TARGET)],
    target=TARGET,
    n_splits=N_SPLITS,
    seed=SEED,
    alpha=10,
)

# Prepare matrices
X = train.drop([TARGET, ID_COL], axis=1)
y = train[TARGET]

test_ids = test[ID_COL]
X_test = test.drop([ID_COL], axis=1)

# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# CatBoost requires categorical features to be strings or integers
# Fill NaNs and convert categorical columns to string to avoid CatBoost errors
if cat_features:
    X[cat_features] = X[cat_features].fillna('missing').astype(str)
    X_test[cat_features] = X_test[cat_features].fillna('missing').astype(str)


# --------------------------------------------------
# STRATIFIED CV
# --------------------------------------------------

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(X_test), N_SPLITS))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n========== Fold {fold + 1} ==========")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=SEED,
        verbose=200,
        early_stopping_rounds=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features
    )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds.reshape(-1)

    test_preds[:, fold] = model.predict(X_test).reshape(-1)

    fold_f1 = f1_score(y_val, val_preds, average="weighted")
    print(f"Fold F1: {fold_f1:.5f}")

print("\nOverall CV F1:",
      f1_score(y, oof_preds, average="weighted"))

# --------------------------------------------------
# FINAL SUBMISSION
# --------------------------------------------------

final_preds = test_preds.mean(axis=1).round().astype(int)
final_preds = le.inverse_transform(final_preds)

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: final_preds
})

submission.to_csv("submission_catboost_baseline.csv", index=False)
print("Submission saved!")
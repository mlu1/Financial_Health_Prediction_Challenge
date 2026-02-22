"""
Tune meta-learner for stacking using OOF meta-features produced from base models.
Saves best meta-learner and tuning results.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from data_loader import load_train_data, load_test_data
from preprocessing import PreprocessingPipeline
import config

print('\nMeta-learner tuning: building OOF meta-features...')

# Load data and pipeline
X, y = load_train_data()

# Encode target variable
le = LabelEncoder()
# Explicitly set the order of classes
ordered_classes = ['Low', 'High', 'Medium']
le.fit(ordered_classes)
y_encoded = le.transform(y)
classes_encoded = np.unique(y_encoded)
print(f"Target classes encoded: {dict(zip(le.classes_, classes_encoded))}")

print("Creating and fitting a new preprocessing pipeline...")
pipeline = PreprocessingPipeline()
X_proc = pipeline.fit_transform(X, y)

print("Saving the updated pipeline...")
with open(config.PREPROCESSING_FILE, 'wb') as f:
    pickle.dump(pipeline, f)

X_ext = X_proc

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=config.RANDOM_SEED, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_SEED)),
    ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_SEED, use_label_encoder=False, eval_metric='mlogloss')),
    ('cat', CatBoostClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_SEED, verbose=0))
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
classes = np.unique(y)

# Define meta-feature columns
base_model_names = [name for name, _ in base_models]
meta_feature_columns = []
for name in base_model_names:
    for c in classes_encoded:
        meta_feature_columns.append(f'{name}_c{c}')
meta_feature_columns.append('cluster')

meta_features = np.zeros((len(X_ext), len(meta_feature_columns)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ext, y_encoded)):
    print(f'Fold {fold+1}/5')
    X_tr, X_val = X_ext.iloc[train_idx], X_ext.iloc[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
    for m, (name, model) in enumerate(base_models):
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)
        for c, cls in enumerate(classes_encoded):
            meta_features[val_idx, m * len(classes_encoded) + c] = probs[:, c]
    meta_features[val_idx, -1] = X_val['cluster'].values

# Save meta-features for inspection
os.makedirs(os.path.join(config.BASE_DIR, 'analysis'), exist_ok=True)
meta_df = pd.DataFrame(meta_features, columns=meta_feature_columns)
meta_df['Target'] = y_encoded
meta_df.to_csv(os.path.join(config.BASE_DIR, 'analysis', 'meta_features_oof.csv'), index=False)
print('Saved OOF meta-features to analysis/meta_features_oof.csv')

# Define candidates for meta-learner
param_grid = [
    {
        'estimator': [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)],
        'estimator__C': [0.01, 0.1, 1.0, 10.0]
    },
    {
        'estimator': [RandomForestClassifier(random_state=config.RANDOM_SEED)],
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [5, 10, None]
    },
    {
        'estimator': [GradientBoostingClassifier(random_state=config.RANDOM_SEED)],
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.01, 0.05, 0.1]
    }
]

# Custom wrapper to allow GridSearch over different estimator classes
from sklearn.base import BaseEstimator, ClassifierMixin, clone
class EstimatorWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=LogisticRegression()):
        self.estimator = estimator
    def fit(self, X, y):
        self.clf_ = clone(self.estimator)
        self.clf_.fit(X, y)
        if hasattr(self.clf_, 'classes_'):
            self.classes_ = self.clf_.classes_
        return self
    def predict(self, X):
        return self.clf_.predict(X)
    def predict_proba(self, X):
        return self.clf_.predict_proba(X)

# Prepare GridSearchCV
X_meta = pd.DataFrame(meta_features, columns=meta_feature_columns)
y_meta = y_encoded
wrapper = EstimatorWrapper()
# Build parameter grid compatible with wrapper
grid_params = []
# Logistic
grid_params.append({
    'estimator': [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)],
    'estimator__C': [0.01, 0.1, 1.0, 10.0]
})
# RandomForest
grid_params.append({
    'estimator': [RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)],
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [5, 10, None]
})
# GradientBoosting
grid_params.append({
    'estimator': [GradientBoostingClassifier(random_state=config.RANDOM_SEED)],
    'estimator__n_estimators': [100, 200],
    'estimator__learning_rate': [0.01, 0.05, 0.1]
})
# LightGBM
grid_params.append({
    'estimator': [lgb.LGBMClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)],
    'estimator__n_estimators': [100, 200],
    'estimator__learning_rate': [0.01, 0.05, 0.1],
    'estimator__num_leaves': [20, 31, 40]
})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(wrapper, grid_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
print('Starting GridSearchCV for meta-learner...')
gs.fit(X_meta, y_meta)

print('Best score:', gs.best_score_)
print('Best params:', gs.best_params_)

# Save best estimator
best_est = gs.best_estimator_.clf_
with open(os.path.join(config.MODEL_DIR, 'meta_learner_tuned.pkl'), 'wb') as f:
    pickle.dump(best_est, f)
print('Saved tuned meta-learner to models/meta_learner_tuned.pkl')

# Save tuning results
results_df = pd.DataFrame(gs.cv_results_)
results_df.to_csv(os.path.join(config.BASE_DIR, 'analysis', 'meta_learner_tuning_results.csv'), index=False)
print('Saved tuning results to analysis/meta_learner_tuning_results.csv')

# Optional: generate stacked submission using tuned meta-learner
print('Generating stacked submission with tuned meta-learner...')
# Re-fit base models on full data and produce test meta-features
from sklearn.cluster import KMeans
X_test, test_ids = load_test_data()
X_test_proc = pipeline.transform(X_test)
X_test_ext = X_test_proc

# Ensure test features match training feature order
X_test_ext = X_test_ext[X_ext.columns]

# Fit base models on full X_ext and predict probs
test_meta = np.zeros((len(X_test_ext), len(meta_feature_columns)))
for m, (name, model) in enumerate(base_models):
    model.fit(X_ext, y_encoded)
    probs = model.predict_proba(X_test_ext)
    for c, cls in enumerate(classes_encoded):
        test_meta[:, m * len(classes_encoded) + c] = probs[:, c]
test_meta[:, -1] = X_test_ext['cluster'].values

test_meta_df = pd.DataFrame(test_meta, columns=meta_feature_columns)
tuned_preds = best_est.predict(test_meta_df)
tuned_preds_decoded = le.inverse_transform(tuned_preds)

# Create submission file
submission = pd.DataFrame({'ID': test_ids, 'Target': tuned_preds_decoded})
submission_path = os.path.join(config.BASE_DIR, 'submissions', 'submission_stacked_tuned.csv')
submission.to_csv(submission_path, index=False)
print(f'Saved tuned stacked submission to {submission_path}')
print('Top 5 rows:\n', submission.head())
print('Class distribution:\n', submission['Target'].value_counts(normalize=True).rename('proportion'))

"""
Configuration file for SME Financial Health Index Challenge
Contains all constants, hyperparameters, and file paths
"""

import os
import random
import numpy as np

# ===========================
# Random Seed for Reproducibility
# ===========================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===========================
# File Paths
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, SUBMISSION_DIR, NOTEBOOKS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "Train.csv")
TEST_FILE = os.path.join(DATA_DIR, "Test.csv")
VARIABLE_DEFS_FILE = os.path.join(DATA_DIR, "VariableDefinitions.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, "SampleSubmission.csv")

# ===========================
# Model & Training Parameters
# ===========================
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

# ===========================
# Feature Engineering
# ===========================
CATEGORICAL_FEATURES = [
    'country',
    'attitude_stable_business_environment',
    'attitude_worried_shutdown',
    'compliance_income_tax',
    'perception_insurance_doesnt_cover_losses',
    'perception_cannot_afford_insurance',
    'motor_vehicle_insurance',
    'has_mobile_money',
    'current_problem_cash_flow',
    'has_cellphone',
    'owner_sex',
    'offers_credit_to_customers',
    'attitude_satisfied_with_achievement',
    'has_credit_card',
    'keeps_financial_records',
    'perception_insurance_companies_dont_insure_businesses_like_yours',
    'perception_insurance_important',
    'has_insurance',
    'covid_essential_service',
    'attitude_more_successful_next_year',
    'problem_sourcing_money',
    'marketing_word_of_mouth',
    'has_loan_account',
    'has_internet_banking',
    'has_debit_card',
    'future_risk_theft_stock',
    'medical_insurance',
    'funeral_insurance',
    'motivation_make_more_money',
    'uses_friends_family_savings',
    'uses_informal_lender',
    # Engineered categorical features
    'business_size',
    'owner_age_group',
    'business_maturity',
    # Statistical categorical features
    'owner_age_quartile',
    'personal_income_quartile',
    'business_expenses_quartile',
    'business_turnover_quartile',
]

# ===========================
# Target encoding settings
# ===========================
# Specify columns to target-encode explicitly (leave empty to auto-detect by cardinality)
TARGET_ENCODING_COLS = []
# Minimum unique values to consider a categorical feature high-cardinality
TARGET_ENCODING_MIN_UNIQUE = 10
# Smoothing parameter for target encoding (higher = stronger shrinkage towards prior)
TARGET_ENCODING_SMOOTH = 5.0

NUMERICAL_FEATURES = [
    'owner_age',
    'personal_income',
    'business_expenses',
    'business_turnover',
    'business_age_years',
    'business_age_months',
    # Engineered financial features
    'profit_margin',
    'expense_ratio', 
    'income_business_ratio',
    'business_efficiency',
    'total_business_months',
    'total_insurance_count',
    'total_banking_services',
    'financial_inclusion_score',
    'positive_attitude_score',
    'risk_awareness_score',
    'age_income_interaction',
    'age_business_size_interaction',
    'business_age_turnover_interaction',
    'gender_income_interaction',
    # Statistical features
    'owner_age_zscore',
    'personal_income_zscore',
    'business_expenses_zscore',
    'business_turnover_zscore',
    'owner_age_percentile',
    'personal_income_percentile',
    'business_expenses_percentile',
    'business_turnover_percentile',
    'owner_age_log',
    'personal_income_log',
    'business_turnover_log',
    'owner_age_sqrt',
    'personal_income_sqrt',
    'business_expenses_sqrt',
    'business_turnover_sqrt',
    'owner_age_outlier_severity',
    'personal_income_outlier_severity',
    'business_expenses_outlier_severity',
    'business_turnover_outlier_severity',
    'income_vs_median',
    'turnover_vs_median',
    'expenses_vs_median',
    'income_stability',
    'cost_efficiency',
    'revenue_per_expense_unit',
    'business_age_ratio',
    'experience_score',
    'age_vs_median',
    'banking_diversity_score',
    'comprehensive_risk_score',
    'compliance_responsibility_score',
]

TARGET = 'Target'
ID_COLUMN = 'ID'

# ===========================
# Preprocessing Parameters
# ===========================
MISSING_VALUE_THRESHOLD = 0.5  # Drop features with >50% missing values
CATEGORICAL_ENCODING = 'label'  # 'label' or 'onehot'

# ===========================
# Output Files
# ===========================
FINAL_MODEL_FILE = os.path.join(MODEL_DIR, "model_final.pkl")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission_final.csv")
PREPROCESSING_FILE = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")

# Use IterativeImputer for numerical features when True (slower, more accurate)
USE_ITERATIVE_IMPUTER = False
# Parameters passed to sklearn.impute.IterativeImputer when enabled
ITERATIVE_IMPUTER_PARAMS = {
    'max_iter': 10,
    'sample_posterior': False,
}

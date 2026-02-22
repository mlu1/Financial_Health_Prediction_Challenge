"""
Preprocessing module for SME Financial Health Index Challenge
Handles data cleaning, encoding, and feature transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    from sklearn.impute import IterativeImputer
except Exception:
    IterativeImputer = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None
from typing import Tuple
import pickle
import sys
import os
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PreprocessingPipeline:
    """
    Preprocessing pipeline for data cleaning and encoding
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.imputers = {}
        self.scaler = StandardScaler()
        self.fitted_features = None  # Track features from training
        self.target_encoding_maps = {}
        self.target_encoding_cols = []
        self.target_classes = []
        self.prior_probs = {}
        
    def create_financial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratio and health indicator features
        """
        X_new = X.copy()
        
        # Financial ratios (safe division)
        turnover_safe = X_new['business_turnover'].fillna(1).replace(0, 1)
        expenses_safe = X_new['business_expenses'].fillna(0)
        income_safe = X_new['personal_income'].fillna(0)
        
        X_new['profit_margin'] = (turnover_safe - expenses_safe) / turnover_safe
        X_new['expense_ratio'] = expenses_safe / turnover_safe
        X_new['income_business_ratio'] = income_safe / turnover_safe
        X_new['business_efficiency'] = turnover_safe / (expenses_safe + 1)
        
        # Financial health indicators
        X_new['has_profit'] = (turnover_safe > expenses_safe).astype(int)
        X_new['high_income'] = (income_safe > income_safe.median()).astype(int)
        X_new['large_business'] = (turnover_safe > turnover_safe.quantile(0.75)).astype(int)
        
        # Business size categories
        business_turnover_filled = X_new['business_turnover'].fillna(0)
        X_new['business_size'] = pd.cut(business_turnover_filled, 
                                      bins=[-np.inf, 1000, 10000, 100000, np.inf], 
                                      labels=['micro', 'small', 'medium', 'large']).astype(str)
        
        return X_new
    
    def create_age_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features
        """
        X_new = X.copy()
        
        # Total business age in months
        X_new['total_business_months'] = X_new['business_age_years'] * 12 + X_new['business_age_months']
        
        # Age categories
        owner_age_filled = X_new['owner_age'].fillna(X_new['owner_age'].median())
        X_new['owner_age_group'] = pd.cut(owner_age_filled, 
                                        bins=[0, 30, 45, 60, 100], 
                                        labels=['young', 'middle', 'senior', 'elderly']).astype(str)
        
        business_months_filled = X_new['total_business_months'].fillna(0)
        X_new['business_maturity'] = pd.cut(business_months_filled, 
                                          bins=[-1, 12, 36, 120, np.inf], 
                                          labels=['new', 'growing', 'established', 'mature']).astype(str)
        
        # Experience indicators
        X_new['experienced_owner'] = (X_new['owner_age'] > 40).astype(int)
        X_new['mature_business'] = (X_new['total_business_months'] > 36).astype(int)
        
        return X_new
    
    def create_insurance_banking_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from insurance and banking columns
        """
        X_new = X.copy()
        
        # Insurance count (how many types of insurance)
        insurance_cols = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance', 'has_insurance']
        for col in insurance_cols:
            if col in X_new.columns:
                X_new[f'{col}_binary'] = (X_new[col] == 'Have now').astype(int)
        
        X_new['total_insurance_count'] = sum([X_new[f'{col}_binary'] for col in insurance_cols if f'{col}_binary' in X_new.columns])
        
        # Banking services count
        banking_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account', 'has_internet_banking', 'has_debit_card']
        for col in banking_cols:
            if col in X_new.columns:
                X_new[f'{col}_binary'] = (X_new[col] == 'Have now').astype(int)
        
        X_new['total_banking_services'] = sum([X_new[f'{col}_binary'] for col in banking_cols if f'{col}_binary' in X_new.columns])
        
        # Financial inclusion score
        X_new['financial_inclusion_score'] = X_new['total_insurance_count'] + X_new['total_banking_services']
        
        return X_new
    
    def create_attitude_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from attitude and perception columns
        """
        X_new = X.copy()
        
        # Positive attitude indicators
        positive_attitudes = [
            'attitude_stable_business_environment',
            'attitude_satisfied_with_achievement', 
            'attitude_more_successful_next_year',
            'perception_insurance_important'
        ]
        
        positive_count = 0
        for col in positive_attitudes:
            if col in X_new.columns:
                X_new[f'{col}_positive'] = (X_new[col] == 'Yes').astype(int)
                positive_count += X_new[f'{col}_positive']
        
        X_new['positive_attitude_score'] = positive_count
        
        # Risk awareness
        risk_cols = ['attitude_worried_shutdown', 'current_problem_cash_flow', 'problem_sourcing_money']
        risk_count = 0
        for col in risk_cols:
            if col in X_new.columns:
                X_new[f'{col}_risk'] = (X_new[col] == 'Yes').astype(int)
                risk_count += X_new[f'{col}_risk']
        
        X_new['risk_awareness_score'] = risk_count
        
        return X_new
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables
        """
        X_new = X.copy()
        
        # Age and financial interactions (safe multiplication)
        owner_age_safe = X_new['owner_age'].fillna(X_new['owner_age'].median())
        income_safe = X_new['personal_income'].fillna(0)
        turnover_safe = X_new['business_turnover'].fillna(1).replace(0, 1)
        total_months_safe = X_new['total_business_months'].fillna(0)
        
        X_new['age_income_interaction'] = owner_age_safe * income_safe
        X_new['age_business_size_interaction'] = owner_age_safe * turnover_safe
        X_new['business_age_turnover_interaction'] = total_months_safe * turnover_safe
        
        # Gender and financial interactions
        if 'owner_sex' in X_new.columns:
            X_new['male_owner'] = (X_new['owner_sex'] == 'Male').astype(int)
            X_new['gender_income_interaction'] = X_new['male_owner'] * income_safe
        else:
            X_new['male_owner'] = 0
            X_new['gender_income_interaction'] = 0
        
        return X_new

    def create_pairwise_numeric_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create all pairwise products and ratios for main numeric features
        """
        X_new = X.copy()
        numeric_cols = [
            'owner_age',
            'personal_income',
            'business_expenses',
            'business_turnover',
            'business_age_years',
            'business_age_months',
            'total_business_months',
        ]
        # Only use columns that exist
        numeric_cols = [col for col in numeric_cols if col in X_new.columns]
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if j <= i:
                    continue  # avoid duplicate pairs and self-pair
                # Products
                X_new[f'{col1}_x_{col2}'] = X_new[col1].fillna(0) * X_new[col2].fillna(0)
                # Ratios (safe division)
                denom = X_new[col2].replace(0, 1).fillna(1)
                X_new[f'{col1}_div_{col2}'] = X_new[col1].fillna(0) / denom
                denom2 = X_new[col1].replace(0, 1).fillna(1)
                X_new[f'{col2}_div_{col1}'] = X_new[col2].fillna(0) / denom2
        return X_new

    def create_groupby_agg_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        For each categorical column, compute mean, min, max of each main numeric column grouped by the categorical column.
        """
        X_new = X.copy()
        # Use config if available, else infer
        if hasattr(config, 'CATEGORICAL_FEATURES'):
            cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in X_new.columns]
        else:
            cat_cols = [c for c in X_new.columns if X_new[c].dtype == 'object' or str(X_new[c].dtype).startswith('category')]
        numeric_cols = [
            'owner_age',
            'personal_income',
            'business_expenses',
            'business_turnover',
            'business_age_years',
            'business_age_months',
            'total_business_months',
        ]
        numeric_cols = [col for col in numeric_cols if col in X_new.columns]
        for cat in cat_cols:
            for num in numeric_cols:
                grp = X_new.groupby(cat)[num]
                mean_map = grp.transform('mean')
                min_map = grp.transform('min')
                max_map = grp.transform('max')
                X_new[f'{cat}_{num}_mean'] = mean_map
                X_new[f'{cat}_{num}_min'] = min_map
                X_new[f'{cat}_{num}_max'] = max_map
        return X_new

    def fit_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit target encoding maps for selected categorical columns.

        For multiclass targets, this creates per-class probability encodings.
        """
        if y is None:
            return

        # Determine columns to encode
        if len(config.TARGET_ENCODING_COLS) > 0:
            cols = [c for c in config.TARGET_ENCODING_COLS if c in X.columns]
        else:
            # Auto-detect high-cardinality object columns
            cols = [c for c in X.columns if (X[c].dtype == 'object' or c in config.CATEGORICAL_FEATURES) and X[c].nunique() >= config.TARGET_ENCODING_MIN_UNIQUE]

        self.target_encoding_cols = cols

        # Compute prior class probabilities
        prior = y.value_counts(normalize=True).to_dict()
        self.prior_probs = prior
        classes = sorted(prior.keys())
        self.target_classes = classes

        k = getattr(config, 'TARGET_ENCODING_SMOOTH', 5.0)
        # --- Build full-data mapping for use at transform time ---
        for col in cols:
            mapping = {}
            grp = pd.DataFrame({col: X[col].astype(str), 'y': y.astype(str)})
            counts = grp.groupby(col)['y'].value_counts().unstack(fill_value=0)
            totals = counts.sum(axis=1)
            for cat in counts.index:
                cat_counts = counts.loc[cat]
                cat_total = totals.loc[cat]
                mapping[cat] = {}
                for cls in classes:
                    class_count = int(cat_counts.get(cls, 0))
                    prob = (class_count + prior.get(cls, 0) * k) / (cat_total + k)
                    mapping[cat][cls] = prob
            self.target_encoding_maps[col] = mapping

        # --- Create OOF encodings for training data to avoid leakage ---
        try:
            from sklearn.model_selection import StratifiedKFold
            n_splits = 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        except Exception:
            skf = None

        self.oof_target_encodings = {}
        if skf is not None:
            for col in cols:
                # Prepare DataFrame to hold per-class oof encodings
                oof_df = pd.DataFrame(index=X.index, columns=[f"{col}_te_{cls}" for cls in classes], dtype=float)
                oof_df[:] = np.nan
                for train_idx, val_idx in skf.split(X, y):
                    train_vals = X.iloc[train_idx][col].astype(str)
                    y_train = y.iloc[train_idx].astype(str)
                    counts = pd.DataFrame({col: train_vals, 'y': y_train}).groupby(col)['y'].value_counts().unstack(fill_value=0)
                    totals = counts.sum(axis=1)
                    mapping_fold = {}
                    for cat in counts.index:
                        mapping_fold[cat] = {}
                        cat_counts = counts.loc[cat]
                        cat_total = totals.loc[cat]
                        for cls in classes:
                            class_count = int(cat_counts.get(cls, 0))
                            mapping_fold[cat][cls] = (class_count + prior.get(cls, 0) * k) / (cat_total + k)

                    # Apply fold mapping to validation indices
                    val_vals = X.iloc[val_idx][col].astype(str)
                    for cls in classes:
                        colname = f"{col}_te_{cls}"
                        oof_df.loc[val_idx, colname] = val_vals.map(lambda v: mapping_fold.get(v, {}).get(cls, prior.get(cls, 0)))

                # Fill any remaining NaNs with prior
                for cls in classes:
                    colname = f"{col}_te_{cls}"
                    oof_df[colname] = oof_df[colname].fillna(prior.get(cls, 0))

                self.oof_target_encodings[col] = oof_df

    def apply_target_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned target-encoding maps to DataFrame, producing per-class numeric columns.

        Drops original column after encoding.
        """
        X_new = X.copy()
        if not self.target_encoding_cols:
            return X_new

        classes = self.target_classes
        prior = self.prior_probs

        for col in self.target_encoding_cols:
            if col not in X_new.columns:
                continue
            col_vals = X_new[col].astype(str)
            mapping = self.target_encoding_maps.get(col, {})
            for cls in classes:
                newcol = f"{col}_te_{cls}"
                X_new[newcol] = col_vals.map(lambda v: mapping.get(v, {}).get(cls, prior.get(cls, 0)))
            # drop original categorical column to avoid double-encoding
            X_new.drop(columns=[col], inplace=True)

        return X_new
    
    def create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features including z-scores, percentiles, and transformations
        """
        X_new = X.copy()
        
        # Numerical columns for statistical analysis
        numerical_cols = ['owner_age', 'personal_income', 'business_expenses', 'business_turnover', 
                         'business_age_years', 'business_age_months', 'total_business_months']
        
        for col in numerical_cols:
            if col in X_new.columns:
                col_data = X_new[col].fillna(X_new[col].median())
                
                # Z-scores (standardization)
                col_mean = col_data.mean()
                col_std = col_data.std()
                if col_std > 0:
                    X_new[f'{col}_zscore'] = (col_data - col_mean) / col_std
                else:
                    X_new[f'{col}_zscore'] = 0
                
                # Percentile rankings
                X_new[f'{col}_percentile'] = col_data.rank(pct=True)
                
                # Log transformations (for positive values) - always create but handle zeros
                col_data_log = col_data.copy()
                col_data_log[col_data_log <= 0] = 1  # Replace non-positive values with 1
                X_new[f'{col}_log'] = np.log1p(col_data_log)
                
                # Square root transformation - always create but handle negatives
                col_data_sqrt = col_data.copy()
                col_data_sqrt[col_data_sqrt < 0] = 0  # Replace negative values with 0
                X_new[f'{col}_sqrt'] = np.sqrt(col_data_sqrt)
                
                # Outlier indicators (values beyond 2 standard deviations)
                if col_std > 0:
                    X_new[f'{col}_is_outlier'] = (np.abs(X_new[f'{col}_zscore']) > 2).astype(int)
                    X_new[f'{col}_outlier_severity'] = np.abs(X_new[f'{col}_zscore'])
                
                # Binned categories (quartiles) - handle edge cases
                try:
                    X_new[f'{col}_quartile'] = pd.qcut(col_data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop').astype(str)
                except (ValueError, TypeError):
                    # Fallback to simple binning if qcut fails
                    X_new[f'{col}_quartile'] = pd.cut(col_data, bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str).fillna('Q1')
        
        return X_new
    
    def create_advanced_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced statistical features including ratios and comparative metrics
        """
        X_new = X.copy()
        
        # Financial comparative features
        if all(col in X_new.columns for col in ['personal_income', 'business_turnover', 'business_expenses']):
            income = X_new['personal_income'].fillna(0)
            turnover = X_new['business_turnover'].fillna(1)
            expenses = X_new['business_expenses'].fillna(0)
            
            # Relative position features (compared to median)
            income_median = income.median()
            turnover_median = turnover.median()
            expenses_median = expenses.median()
            
            X_new['income_vs_median'] = income / (income_median + 1)
            X_new['turnover_vs_median'] = turnover / (turnover_median + 1)
            X_new['expenses_vs_median'] = expenses / (expenses_median + 1)
            
            # Financial stability indicators
            X_new['income_stability'] = np.minimum(income / (turnover + 1), 2)  # Cap at 2
            X_new['cost_efficiency'] = expenses / (turnover + 1)
            X_new['revenue_per_expense_unit'] = turnover / (expenses + 1)
            
        # Age-related statistical features
        if all(col in X_new.columns for col in ['owner_age', 'total_business_months']):
            age = X_new['owner_age'].fillna(X_new['owner_age'].median())
            business_months = X_new['total_business_months'].fillna(0)
            
            # Experience ratios
            X_new['business_age_ratio'] = business_months / (age * 12 + 1)  # What fraction of life spent in business
            X_new['experience_score'] = np.sqrt(age) * np.log1p(business_months)  # Combined experience metric
            
            # Age group statistical features
            age_median = age.median()
            X_new['age_vs_median'] = age / age_median
            X_new['young_entrepreneur'] = ((age < 30) & (business_months > 12)).astype(int)
            X_new['senior_entrepreneur'] = ((age > 55) & (business_months > 60)).astype(int)
        
        return X_new
    
    def create_distribution_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on data distribution characteristics
        """
        X_new = X.copy()
        
        # Create portfolio diversity scores based on categorical features
        categorical_diversity_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account', 
                                    'has_internet_banking', 'has_debit_card']
        
        diversity_count = 0
        for col in categorical_diversity_cols:
            if col in X_new.columns:
                # Count 'Have now' responses
                has_service = (X_new[col] == 'Have now').astype(int)
                diversity_count += has_service
        
        X_new['banking_diversity_score'] = diversity_count
        X_new['is_banking_diverse'] = (diversity_count >= 3).astype(int)
        
        # Risk profile based on multiple risk indicators
        risk_cols = ['attitude_worried_shutdown', 'current_problem_cash_flow', 'problem_sourcing_money']
        risk_score = 0
        for col in risk_cols:
            if col in X_new.columns:
                has_risk = (X_new[col] == 'Yes').astype(int)
                risk_score += has_risk
        
        X_new['comprehensive_risk_score'] = risk_score
        X_new['is_high_risk'] = (risk_score >= 2).astype(int)
        
        # Compliance and responsibility score
        compliance_cols = ['compliance_income_tax', 'keeps_financial_records']
        compliance_score = 0
        for col in compliance_cols:
            if col in X_new.columns:
                is_compliant = (X_new[col] == 'Yes').astype(int)
                compliance_score += is_compliant
        
        X_new['compliance_responsibility_score'] = compliance_score
        X_new['is_highly_compliant'] = (compliance_score == 2).astype(int)
        
        return X_new

    def create_cluster_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Create cluster features using KMeans on numeric columns.
        """
        X_new = X.copy()
        if KMeans is None:
            return X_new
        # Choose numeric columns (exclude any existing cluster column)
        numeric_cols = [c for c in X_new.columns if pd.api.types.is_numeric_dtype(X_new[c]) and c != 'cluster']

        if not numeric_cols:
            return X_new

        # On fit: record numeric columns used, fit a scaler then fit KMeans on scaled data
        if fit:
            self.cluster_numeric_cols = numeric_cols
            # Create and fit a scaler specifically for clustering
            from sklearn.preprocessing import StandardScaler as _StandardScaler
            self.kmeans_scaler = _StandardScaler()
            try:
                X_scaled = self.kmeans_scaler.fit_transform(X_new[self.cluster_numeric_cols].fillna(0))
            except Exception:
                # Fallback: coerce to numeric and fillna
                X_scaled = self.kmeans_scaler.fit_transform(X_new[self.cluster_numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0))

            # Fit KMeans on scaled features
            self.kmeans = KMeans(n_clusters=getattr(config, 'CLUSTER_N_CLUSTERS', 4),
                                  random_state=config.RANDOM_SEED,
                                  n_init=getattr(config, 'CLUSTER_N_INIT', 10))
            labels = self.kmeans.fit_predict(X_scaled)
            X_new['cluster'] = labels

        else:
            # On transform: require fitted scaler and numeric column list
            if not hasattr(self, 'kmeans') or not hasattr(self, 'kmeans_scaler') or not hasattr(self, 'cluster_numeric_cols'):
                return X_new

            # Ensure all expected cluster numeric cols exist in incoming data
            cols = [c for c in self.cluster_numeric_cols if c in X_new.columns]
            if not cols:
                return X_new

            try:
                X_scaled = self.kmeans_scaler.transform(X_new[cols].fillna(0))
            except Exception:
                X_scaled = self.kmeans_scaler.transform(X_new[cols].apply(pd.to_numeric, errors='coerce').fillna(0))

            labels = self.kmeans.predict(X_scaled)
            X_new['cluster'] = labels

        return X_new
        
    def handle_missing_values(self, X: pd.DataFrame, fit: bool = True, strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in features
        
        Args:
            X: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
        Returns:
            DataFrame with missing values handled
        """
        X_processed = X.copy()

        # Fast fallback imputation (median/mode) unless iterative imputer enabled
        if getattr(config, 'USE_ITERATIVE_IMPUTER', False) and IterativeImputer is not None:
            # Fill categorical with mode first (iterative imputer handles numerical only)
            for col in config.CATEGORICAL_FEATURES:
                if col in X_processed.columns:
                    mode_val = X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'Unknown'
                    X_processed[col] = X_processed[col].fillna(mode_val)

            # Identify numerical columns available in this dataset
            num_cols = [c for c in config.NUMERICAL_FEATURES if c in X_processed.columns]

            if fit:
                # Fit iterative imputer on available numerical columns
                if 'iterative' not in self.imputers or self.imputers.get('iterative') is None:
                    imputer_params = getattr(config, 'ITERATIVE_IMPUTER_PARAMS', {}) or {}
                    imputer = IterativeImputer(**imputer_params, random_state=getattr(config, 'RANDOM_SEED', None))
                    try:
                        imputed = imputer.fit_transform(X_processed[num_cols])
                        self.imputers['iterative'] = imputer
                        X_processed[num_cols] = imputed
                    except Exception:
                        for col in num_cols:
                            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else:
                    try:
                        imputed = self.imputers['iterative'].fit_transform(X_processed[num_cols])
                        X_processed[num_cols] = imputed
                    except Exception:
                        for col in num_cols:
                            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            else:
                # Use fitted imputer if available; otherwise fallback to simple median
                if 'iterative' in self.imputers and self.imputers.get('iterative') is not None:
                    try:
                        # Ensure only columns present in the test set are used
                        num_cols_present = [c for c in num_cols if c in X_processed.columns]
                        if num_cols_present:
                            imputed = self.imputers['iterative'].transform(X_processed[num_cols_present])
                            X_processed[num_cols_present] = imputed
                    except Exception:
                        # Fallback for all numeric columns on error
                        for col in num_cols:
                            if col in X_processed.columns:
                                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else:
                    # Fallback if no iterative imputer is fitted
                    for col in num_cols:
                        if col in X_processed.columns:
                            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

            return X_processed

        # Default simple imputation: median for numeric, mode for categorical
        for col in config.NUMERICAL_FEATURES:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())

        for col in config.CATEGORICAL_FEATURES:
            if col in X_processed.columns:
                mode_val = X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'Unknown'
                X_processed[col] = X_processed[col].fillna(mode_val)

        return X_processed
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder
        
        Args:
            X: Input DataFrame
            fit: If True, fit the encoders; otherwise use existing
        
        Returns:
            DataFrame with encoded categorical features
        """
        X_encoded = X.copy()
        
        # Get all categorical columns (both from config and dynamically detected)
        categorical_cols = set(config.CATEGORICAL_FEATURES)
        
        # Add dynamically detected categorical columns (object/string columns)
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                categorical_cols.add(col)
        
        for col in categorical_cols:
            if col not in X_encoded.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                # Convert to string first to handle mixed types
                col_values = X_encoded[col].astype(str)
                X_encoded[col] = self.label_encoders[col].fit_transform(col_values)
            else:
                if col in self.label_encoders:
                    # Handle unseen categories by assigning to a default category
                    col_values = X_encoded[col].astype(str)
                    # Get known classes
                    known_classes = set(self.label_encoders[col].classes_)
                    # Replace unseen values with a known value (first class or most common during training)
                    col_values = col_values.apply(lambda x: x if x in known_classes else self.label_encoders[col].classes_[0])
                    X_encoded[col] = self.label_encoders[col].transform(col_values)
        
        return X_encoded
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform data
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        X_processed = self.handle_missing_values(X)
        if y is not None:
            self.fit_target_encoding(X_processed, y)
            if hasattr(self, 'oof_target_encodings') and self.oof_target_encodings:
                for col, oof_df in self.oof_target_encodings.items():
                    X_processed = pd.concat([X_processed, oof_df], axis=1)
                    if col in X_processed.columns:
                        X_processed.drop(columns=[col], inplace=True)
            else:
                X_processed = self.apply_target_encoding(X_processed)
        X_processed = self.create_financial_features(X_processed)
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_insurance_banking_features(X_processed)
        X_processed = self.create_attitude_features(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_pairwise_numeric_interactions(X_processed)
        X_processed = self.create_groupby_agg_features(X_processed)
        X_processed = self.create_statistical_features(X_processed)
        X_processed = self.create_advanced_statistical_features(X_processed)
        X_processed = self.create_distribution_features(X_processed)

        # Final encoding of any remaining categorical features
        X_processed = self.encode_categorical_features(X_processed, fit=True)

        # Fill any remaining NaNs after all feature creation
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                 X_processed[col] = X_processed[col].astype('category').cat.codes
            X_processed[col] = X_processed[col].fillna(0)

        # Clustering must be the final step on fully numeric data
        X_processed = self.create_cluster_features(X_processed, fit=True)

        self.fitted_features = list(X_processed.columns)
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        X_processed = self.handle_missing_values(X, fit=False)
        X_processed = self.apply_target_encoding(X_processed)
        X_processed = self.create_financial_features(X_processed)
        X_processed = self.create_age_features(X_processed)
        X_processed = self.create_insurance_banking_features(X_processed)
        X_processed = self.create_attitude_features(X_processed)
        X_processed = self.create_interaction_features(X_processed)
        X_processed = self.create_pairwise_numeric_interactions(X_processed)
        X_processed = self.create_groupby_agg_features(X_processed)
        X_processed = self.create_statistical_features(X_processed)
        X_processed = self.create_advanced_statistical_features(X_processed)
        X_processed = self.create_distribution_features(X_processed)

        # Final encoding of any remaining categorical features
        X_processed = self.encode_categorical_features(X_processed, fit=False)

        # Fill any remaining NaNs after all feature creation
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                 X_processed[col] = X_processed[col].astype('category').cat.codes
            X_processed[col] = X_processed[col].fillna(0)

        # Clustering must be the final step on fully numeric data
        X_processed = self.create_cluster_features(X_processed, fit=False)

        # Align columns with fitted features
        missing_cols = set(self.fitted_features) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        
        X_processed = X_processed[self.fitted_features]
        
        return X_processed


def preprocess_data(X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Main preprocessing function
    
    Args:
        X: Features DataFrame
        y: Target Series (optional)
        fit: If True, fit the preprocessing; otherwise use existing
    
    Returns:
        Tuple of (preprocessed X, y)
    """
    if fit:
        pipeline = PreprocessingPipeline()
        X_processed = pipeline.fit_transform(X, y)
        # Save pipeline
        with open(config.PREPROCESSING_FILE, 'wb') as f:
            pickle.dump(pipeline, f)
    else:
        # Load previously saved pipeline and apply transform
        try:
            pipeline = load_preprocessing_pipeline()
        except Exception:
            pipeline = PreprocessingPipeline()
        X_processed = pipeline.transform(X)
    
    return X_processed, y


def save_preprocessing_pipeline(pipeline: PreprocessingPipeline) -> None:
    """Save preprocessing pipeline to disk"""
    with open(config.PREPROCESSING_FILE, 'wb') as f:
        pickle.dump(pipeline, f)


def load_preprocessing_pipeline() -> PreprocessingPipeline:
    """Load preprocessing pipeline from disk"""
    with open(config.PREPROCESSING_FILE, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_loader import load_train_data, load_test_data
    
    X_train, y_train = load_train_data()
    X_test, test_ids = load_test_data()
    
    print("Original data shape:", X_train.shape)
    
    X_train_processed, y_train = preprocess_data(X_train, y_train, fit=True)
    print("Processed train data shape:", X_train_processed.shape)
    
    X_test_processed, test_ids = preprocess_data(X_test, fit=False)
    print("Processed test data shape:", X_test_processed.shape)


"""
Preprocessing module for SME Financial Health Index Challenge
Handles data cleaning, encoding, and feature transformation

Key fixes vs previous version:
- All statistics used in feature engineering (medians/means/std/quantiles) are FIT on train and REUSED on transform.
- Groupby aggregate features are built from TRAIN mappings, then applied via map() on val/test.
- Missing-value medians/modes are FIT once and reused.
- IterativeImputer: transform() uses transform (NOT fit_transform).
- Clustering already was fit/transform safe; kept and hardened.
"""

import os
import sys
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from sklearn.impute import IterativeImputer
except Exception:
    IterativeImputer = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PreprocessingPipeline:
    """
    Preprocessing pipeline for data cleaning and encoding.
    """

    def __init__(self):
        # encoders / imputers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.fitted_features = None  # feature list from training

        # target encoding
        self.target_encoding_maps = {}
        self.target_encoding_cols = []
        self.target_classes = []
        self.prior_probs = {}

        # fit-time state for consistency
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, Any] = {}

        # stats used by feature engineering
        self.feature_stats_: Dict[str, Dict[str, float]] = {}          # {col: {median, mean, std}}
        self.feature_quantile_bins_: Dict[str, list] = {}              # {col: [q0,q25,q50,q75,q100]}
        self.feature_sorted_values_: Dict[str, np.ndarray] = {}        # {col: sorted train vals}
        self.financial_thresholds_: Dict[str, float] = {}              # income_median, turnover_q75, etc.

        # groupby agg mappings
        self.groupby_maps_: Dict[tuple, dict] = {}                     # key=(cat,num,agg)
        self.groupby_fallback_: Dict[str, Dict[str, float]] = {}       # num -> {mean/min/max fallback}

    # --------------------------
    # Fit-time utilities
    # --------------------------
    def _fit_feature_stats(self, X: pd.DataFrame) -> None:
        """Fit and store global statistics used by feature-engineering steps."""
        numerical_cols = [
            "owner_age",
            "personal_income",
            "business_expenses",
            "business_turnover",
            "business_age_years",
            "business_age_months",
            "total_business_months",
        ]

        for col in numerical_cols:
            if col not in X.columns:
                continue
            s = pd.to_numeric(X[col], errors="coerce")
            med = float(s.median()) if np.isfinite(s.median()) else 0.0
            mean = float(s.mean()) if np.isfinite(s.mean()) else med
            std = float(s.std()) if np.isfinite(s.std()) else 0.0

            self.feature_stats_[col] = {"median": med, "mean": mean, "std": std}

            qs = s.dropna()
            if len(qs) >= 4:
                edges = qs.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).astype(float).tolist()
                # keep only if we have enough unique edges to form bins
                if all(np.isfinite(edges)) and len(set(edges)) >= 3:
                    self.feature_quantile_bins_[col] = edges

            if len(qs) > 0:
                self.feature_sorted_values_[col] = np.sort(qs.values)

        # thresholds for indicator features
        if "personal_income" in X.columns:
            self.financial_thresholds_["income_median"] = float(
                pd.to_numeric(X["personal_income"], errors="coerce").median()
            )
        if "business_turnover" in X.columns:
            t = pd.to_numeric(X["business_turnover"], errors="coerce")
            self.financial_thresholds_["turnover_q75"] = float(t.quantile(0.75))
            self.financial_thresholds_["turnover_median"] = float(t.median())
        if "business_expenses" in X.columns:
            self.financial_thresholds_["expenses_median"] = float(
                pd.to_numeric(X["business_expenses"], errors="coerce").median()
            )
        if "owner_age" in X.columns:
            self.financial_thresholds_["owner_age_median"] = float(
                pd.to_numeric(X["owner_age"], errors="coerce").median()
            )

    def _percentile_from_sorted(self, sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Approximate percentile (0..1) based on training distribution."""
        if sorted_vals is None or len(sorted_vals) == 0:
            return np.zeros_like(x, dtype=float)
        idx = np.searchsorted(sorted_vals, x, side="right")
        return idx / float(len(sorted_vals))

    # --------------------------
    # Feature blocks (fit/transform safe)
    # --------------------------
    def create_financial_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create financial ratio and health indicator features (thresholds fit on train)."""
        X_new = X.copy()
        if "business_turnover" not in X_new.columns:
            return X_new

        turnover = pd.to_numeric(X_new["business_turnover"], errors="coerce")
        expenses = (
            pd.to_numeric(X_new["business_expenses"], errors="coerce")
            if "business_expenses" in X_new.columns
            else pd.Series(0.0, index=X_new.index)
        )
        income = (
            pd.to_numeric(X_new["personal_income"], errors="coerce")
            if "personal_income" in X_new.columns
            else pd.Series(0.0, index=X_new.index)
        )

        turnover_safe = turnover.fillna(1).replace(0, 1)
        expenses_safe = expenses.fillna(0)
        income_safe = income.fillna(0)

        # Ratios
        X_new["profit_margin"] = (turnover_safe - expenses_safe) / turnover_safe
        X_new["expense_ratio"] = expenses_safe / turnover_safe
        X_new["income_business_ratio"] = income_safe / turnover_safe
        X_new["business_efficiency"] = turnover_safe / (expenses_safe + 1)

        if fit:
            self.financial_thresholds_["income_median"] = float(income_safe.median())
            self.financial_thresholds_["turnover_q75"] = float(turnover_safe.quantile(0.75))

        income_med = self.financial_thresholds_.get("income_median", float(income_safe.median()))
        turnover_q75 = self.financial_thresholds_.get("turnover_q75", float(turnover_safe.quantile(0.75)))

        X_new["has_profit"] = (turnover_safe > expenses_safe).astype(int)
        X_new["high_income"] = (income_safe > income_med).astype(int)
        X_new["large_business"] = (turnover_safe > turnover_q75).astype(int)

        # Stable binning (fixed edges)
        X_new["business_size"] = pd.cut(
            turnover.fillna(0),
            bins=[-np.inf, 1000, 10000, 100000, np.inf],
            labels=["micro", "small", "medium", "large"],
        ).astype(str)

        return X_new

    def create_age_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create age-related features (medians learned on train)."""
        X_new = X.copy()

        if "business_age_years" in X_new.columns and "business_age_months" in X_new.columns:
            X_new["total_business_months"] = (
                pd.to_numeric(X_new["business_age_years"], errors="coerce").fillna(0) * 12
                + pd.to_numeric(X_new["business_age_months"], errors="coerce").fillna(0)
            )

        if "owner_age" in X_new.columns:
            owner_age = pd.to_numeric(X_new["owner_age"], errors="coerce")
            if fit:
                self.financial_thresholds_["owner_age_median"] = float(owner_age.median())
            owner_age_med = self.financial_thresholds_.get("owner_age_median", float(owner_age.median()))
            owner_age_filled = owner_age.fillna(owner_age_med)

            X_new["owner_age_group"] = pd.cut(
                owner_age_filled,
                bins=[0, 30, 45, 60, 100],
                labels=["young", "middle", "senior", "elderly"],
            ).astype(str)

            X_new["experienced_owner"] = (owner_age_filled > 40).astype(int)

        if "total_business_months" in X_new.columns:
            business_months = pd.to_numeric(X_new["total_business_months"], errors="coerce").fillna(0)
            X_new["business_maturity"] = pd.cut(
                business_months,
                bins=[-1, 12, 36, 120, np.inf],
                labels=["new", "growing", "established", "mature"],
            ).astype(str)

            X_new["mature_business"] = (business_months > 36).astype(int)

        return X_new

    def create_insurance_banking_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features from insurance and banking columns."""
        X_new = X.copy()

        insurance_cols = ["motor_vehicle_insurance", "medical_insurance", "funeral_insurance", "has_insurance"]
        for col in insurance_cols:
            if col in X_new.columns:
                X_new[f"{col}_binary"] = (X_new[col] == "Have now").astype(int)
        X_new["total_insurance_count"] = sum(
            [X_new[f"{col}_binary"] for col in insurance_cols if f"{col}_binary" in X_new.columns]
        )

        banking_cols = ["has_mobile_money", "has_credit_card", "has_loan_account", "has_internet_banking", "has_debit_card"]
        for col in banking_cols:
            if col in X_new.columns:
                X_new[f"{col}_binary"] = (X_new[col] == "Have now").astype(int)
        X_new["total_banking_services"] = sum(
            [X_new[f"{col}_binary"] for col in banking_cols if f"{col}_binary" in X_new.columns]
        )

        X_new["financial_inclusion_score"] = X_new["total_insurance_count"] + X_new["total_banking_services"]
        return X_new

    def create_attitude_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features from attitude and perception columns."""
        X_new = X.copy()

        positive_attitudes = [
            "attitude_stable_business_environment",
            "attitude_satisfied_with_achievement",
            "attitude_more_successful_next_year",
            "perception_insurance_important",
        ]
        positive_count = 0
        for col in positive_attitudes:
            if col in X_new.columns:
                X_new[f"{col}_positive"] = (X_new[col] == "Yes").astype(int)
                positive_count += X_new[f"{col}_positive"]
        X_new["positive_attitude_score"] = positive_count

        risk_cols = ["attitude_worried_shutdown", "current_problem_cash_flow", "problem_sourcing_money"]
        risk_count = 0
        for col in risk_cols:
            if col in X_new.columns:
                X_new[f"{col}_risk"] = (X_new[col] == "Yes").astype(int)
                risk_count += X_new[f"{col}_risk"]
        X_new["risk_awareness_score"] = risk_count

        return X_new

    def create_interaction_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create interaction features between key variables (uses train medians)."""
        X_new = X.copy()

        owner_age = pd.to_numeric(X_new["owner_age"], errors="coerce") if "owner_age" in X_new.columns else pd.Series(0.0, index=X_new.index)
        income = pd.to_numeric(X_new["personal_income"], errors="coerce") if "personal_income" in X_new.columns else pd.Series(0.0, index=X_new.index)
        turnover = pd.to_numeric(X_new["business_turnover"], errors="coerce") if "business_turnover" in X_new.columns else pd.Series(1.0, index=X_new.index)
        total_months = pd.to_numeric(X_new["total_business_months"], errors="coerce") if "total_business_months" in X_new.columns else pd.Series(0.0, index=X_new.index)

        if fit and "owner_age_median" not in self.financial_thresholds_:
            self.financial_thresholds_["owner_age_median"] = float(owner_age.median())

        owner_age_med = self.financial_thresholds_.get("owner_age_median", float(owner_age.median()))
        owner_age_safe = owner_age.fillna(owner_age_med)
        income_safe = income.fillna(0)
        turnover_safe = turnover.fillna(1).replace(0, 1)
        total_months_safe = total_months.fillna(0)

        X_new["age_income_interaction"] = owner_age_safe * income_safe
        X_new["age_business_size_interaction"] = owner_age_safe * turnover_safe
        X_new["business_age_turnover_interaction"] = total_months_safe * turnover_safe

        if "owner_sex" in X_new.columns:
            X_new["male_owner"] = (X_new["owner_sex"] == "Male").astype(int)
            X_new["gender_income_interaction"] = X_new["male_owner"] * income_safe
        else:
            X_new["male_owner"] = 0
            X_new["gender_income_interaction"] = 0

        return X_new

    def create_pairwise_numeric_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create all pairwise products and ratios for main numeric features."""
        X_new = X.copy()
        numeric_cols = [
            "owner_age",
            "personal_income",
            "business_expenses",
            "business_turnover",
            "business_age_years",
            "business_age_months",
            "total_business_months",
        ]
        numeric_cols = [col for col in numeric_cols if col in X_new.columns]

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if j <= i:
                    continue
                X_new[f"{col1}_x_{col2}"] = pd.to_numeric(X_new[col1], errors="coerce").fillna(0) * pd.to_numeric(X_new[col2], errors="coerce").fillna(0)

                denom = pd.to_numeric(X_new[col2], errors="coerce").replace(0, 1).fillna(1)
                X_new[f"{col1}_div_{col2}"] = pd.to_numeric(X_new[col1], errors="coerce").fillna(0) / denom

                denom2 = pd.to_numeric(X_new[col1], errors="coerce").replace(0, 1).fillna(1)
                X_new[f"{col2}_div_{col1}"] = pd.to_numeric(X_new[col2], errors="coerce").fillna(0) / denom2

        return X_new

    def create_groupby_agg_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        For each categorical column, compute mean/min/max of numeric columns grouped by category.
        FIX: Fit mappings on TRAIN only; apply mappings on transform.
        """
        X_new = X.copy()

        if hasattr(config, "CATEGORICAL_FEATURES"):
            cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in X_new.columns]
        else:
            cat_cols = [c for c in X_new.columns if X_new[c].dtype == "object" or str(X_new[c].dtype).startswith("category")]

        numeric_cols = [
            "owner_age",
            "personal_income",
            "business_expenses",
            "business_turnover",
            "business_age_years",
            "business_age_months",
            "total_business_months",
        ]
        numeric_cols = [c for c in numeric_cols if c in X_new.columns]

        if fit:
            self.groupby_maps_.clear()
            self.groupby_fallback_.clear()

            for num in numeric_cols:
                s = pd.to_numeric(X_new[num], errors="coerce")
                self.groupby_fallback_[num] = {
                    "mean": float(s.mean()) if np.isfinite(s.mean()) else 0.0,
                    "min": float(s.min()) if np.isfinite(s.min()) else 0.0,
                    "max": float(s.max()) if np.isfinite(s.max()) else 0.0,
                }

            for cat in cat_cols:
                cat_s = X_new[cat].astype(str)
                for num in numeric_cols:
                    num_s = pd.to_numeric(X_new[num], errors="coerce")
                    df = pd.DataFrame({"cat": cat_s, "num": num_s})
                    grp = df.groupby("cat")["num"]
                    self.groupby_maps_[(cat, num, "mean")] = grp.mean().to_dict()
                    self.groupby_maps_[(cat, num, "min")] = grp.min().to_dict()
                    self.groupby_maps_[(cat, num, "max")] = grp.max().to_dict()

        # apply
        if not getattr(self, "groupby_maps_", None):
            return X_new

        for cat in cat_cols:
            cat_s = X_new[cat].astype(str)
            for num in numeric_cols:
                for agg in ("mean", "min", "max"):
                    mapping = self.groupby_maps_.get((cat, num, agg), {})
                    fallback = self.groupby_fallback_.get(num, {}).get(agg, 0.0)
                    X_new[f"{cat}_{num}_{agg}"] = cat_s.map(mapping).fillna(fallback)

        return X_new

    # --------------------------
    # Target encoding
    # --------------------------
    def fit_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit target encoding maps (multiclass) and OOF encodings for train."""
        if y is None:
            return

        if len(getattr(config, "TARGET_ENCODING_COLS", [])) > 0:
            cols = [c for c in config.TARGET_ENCODING_COLS if c in X.columns]
        else:
            cols = [
                c
                for c in X.columns
                if (X[c].dtype == "object" or c in getattr(config, "CATEGORICAL_FEATURES", []))
                and X[c].nunique() >= getattr(config, "TARGET_ENCODING_MIN_UNIQUE", 10)
            ]

        self.target_encoding_cols = cols
        prior = y.value_counts(normalize=True).to_dict()
        self.prior_probs = prior
        classes = sorted(prior.keys())
        self.target_classes = classes
        k = float(getattr(config, "TARGET_ENCODING_SMOOTH", 5.0))

        # full mapping (used for transform)
        self.target_encoding_maps = {}
        for col in cols:
            mapping = {}
            grp = pd.DataFrame({col: X[col].astype(str), "y": y.astype(str)})
            counts = grp.groupby(col)["y"].value_counts().unstack(fill_value=0)
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

        # OOF encodings (train only) to reduce leakage
        self.oof_target_encodings = {}
        try:
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=getattr(config, "RANDOM_SEED", 42))
        except Exception:
            skf = None

        if skf is not None and len(cols) > 0:
            y_str = y.astype(str)
            for col in cols:
                oof_df = pd.DataFrame(index=X.index, columns=[f"{col}_te_{cls}" for cls in classes], dtype=float)
                oof_df[:] = np.nan

                for tr_idx, va_idx in skf.split(X, y_str):
                    X_tr_col = X.iloc[tr_idx][col].astype(str)
                    y_tr = y_str.iloc[tr_idx]

                    counts = (
                        pd.DataFrame({col: X_tr_col, "y": y_tr})
                        .groupby(col)["y"]
                        .value_counts()
                        .unstack(fill_value=0)
                    )
                    totals = counts.sum(axis=1)

                    mapping_fold = {}
                    for cat in counts.index:
                        mapping_fold[cat] = {}
                        cat_counts = counts.loc[cat]
                        cat_total = totals.loc[cat]
                        for cls in classes:
                            class_count = int(cat_counts.get(cls, 0))
                            mapping_fold[cat][cls] = (class_count + prior.get(cls, 0) * k) / (cat_total + k)

                    X_va_col = X.iloc[va_idx][col].astype(str)
                    for cls in classes:
                        oof_df.loc[va_idx, f"{col}_te_{cls}"] = X_va_col.map(
                            lambda v: mapping_fold.get(v, {}).get(cls, prior.get(cls, 0))
                        )

                for cls in classes:
                    oof_df[f"{col}_te_{cls}"] = oof_df[f"{col}_te_{cls}"].fillna(prior.get(cls, 0))

                self.oof_target_encodings[col] = oof_df

    def apply_target_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned target encoding maps to X and drop original encoded columns."""
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

            X_new.drop(columns=[col], inplace=True)

        return X_new

    # --------------------------
    # Statistical features (fit/transform safe)
    # --------------------------
    def create_statistical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create statistical features using train-fitted stats (zscore/percentile/quartile)."""
        X_new = X.copy()

        numerical_cols = [
            "owner_age",
            "personal_income",
            "business_expenses",
            "business_turnover",
            "business_age_years",
            "business_age_months",
            "total_business_months",
        ]

        for col in numerical_cols:
            if col not in X_new.columns:
                continue

            s = pd.to_numeric(X_new[col], errors="coerce")

            # fill using train median
            med = self.feature_stats_.get(col, {}).get("median", float(s.median()) if np.isfinite(s.median()) else 0.0)
            col_data = s.fillna(med)

            # zscore using train mean/std
            mean = self.feature_stats_.get(col, {}).get("mean", float(col_data.mean()))
            std = self.feature_stats_.get(col, {}).get("std", float(col_data.std()))
            if std and std > 0:
                X_new[f"{col}_zscore"] = (col_data - mean) / std
            else:
                X_new[f"{col}_zscore"] = 0.0

            # percentile based on train distribution
            sorted_vals = self.feature_sorted_values_.get(col)
            X_new[f"{col}_percentile"] = self._percentile_from_sorted(sorted_vals, col_data.to_numpy())

            # log and sqrt (stable transforms)
            col_data_log = col_data.copy()
            col_data_log[col_data_log <= 0] = 1
            X_new[f"{col}_log"] = np.log1p(col_data_log)

            col_data_sqrt = col_data.copy()
            col_data_sqrt[col_data_sqrt < 0] = 0
            X_new[f"{col}_sqrt"] = np.sqrt(col_data_sqrt)

            # outlier indicators (relative to train zscore)
            X_new[f"{col}_is_outlier"] = (np.abs(X_new[f"{col}_zscore"]) > 2).astype(int)
            X_new[f"{col}_outlier_severity"] = np.abs(X_new[f"{col}_zscore"])

            # quartiles using train quantile edges (fallback if degenerate)
            edges = self.feature_quantile_bins_.get(col)
            if edges is not None and len(edges) == 5 and len(set(edges)) >= 3:
                # pad bins slightly if min == max (rare), otherwise pd.cut may error
                try:
                    X_new[f"{col}_quartile"] = pd.cut(
                        col_data,
                        bins=edges,
                        labels=["Q1", "Q2", "Q3", "Q4"],
                        include_lowest=True,
                        duplicates="drop",
                    ).astype(str)
                except Exception:
                    X_new[f"{col}_quartile"] = "Q1"
            else:
                # fallback: fixed-width binning
                try:
                    X_new[f"{col}_quartile"] = pd.cut(col_data, bins=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str).fillna("Q1")
                except Exception:
                    X_new[f"{col}_quartile"] = "Q1"

        return X_new

    def create_advanced_statistical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create advanced statistical features using train-fitted medians."""
        X_new = X.copy()

        if all(c in X_new.columns for c in ["personal_income", "business_turnover", "business_expenses"]):
            income = pd.to_numeric(X_new["personal_income"], errors="coerce").fillna(0)
            turnover = pd.to_numeric(X_new["business_turnover"], errors="coerce").fillna(1)
            expenses = pd.to_numeric(X_new["business_expenses"], errors="coerce").fillna(0)

            if fit:
                self.financial_thresholds_["income_median"] = float(income.median())
                self.financial_thresholds_["turnover_median"] = float(turnover.median())
                self.financial_thresholds_["expenses_median"] = float(expenses.median())

            income_median = self.financial_thresholds_.get("income_median", float(income.median()))
            turnover_median = self.financial_thresholds_.get("turnover_median", float(turnover.median()))
            expenses_median = self.financial_thresholds_.get("expenses_median", float(expenses.median()))

            X_new["income_vs_median"] = income / (income_median + 1)
            X_new["turnover_vs_median"] = turnover / (turnover_median + 1)
            X_new["expenses_vs_median"] = expenses / (expenses_median + 1)

            X_new["income_stability"] = np.minimum(income / (turnover + 1), 2)
            X_new["cost_efficiency"] = expenses / (turnover + 1)
            X_new["revenue_per_expense_unit"] = turnover / (expenses + 1)

        if all(c in X_new.columns for c in ["owner_age", "total_business_months"]):
            age = pd.to_numeric(X_new["owner_age"], errors="coerce")
            if fit and "owner_age_median" not in self.financial_thresholds_:
                self.financial_thresholds_["owner_age_median"] = float(age.median())
            age_med = self.financial_thresholds_.get("owner_age_median", float(age.median()))
            age = age.fillna(age_med)

            business_months = pd.to_numeric(X_new["total_business_months"], errors="coerce").fillna(0)

            X_new["business_age_ratio"] = business_months / (age * 12 + 1)
            X_new["experience_score"] = np.sqrt(age) * np.log1p(business_months)
            X_new["age_vs_median"] = age / (age_med + 1e-9)

            X_new["young_entrepreneur"] = ((age < 30) & (business_months > 12)).astype(int)
            X_new["senior_entrepreneur"] = ((age > 55) & (business_months > 60)).astype(int)

        return X_new

    def create_distribution_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create simple distribution/profile features."""
        X_new = X.copy()

        categorical_diversity_cols = [
            "has_mobile_money",
            "has_credit_card",
            "has_loan_account",
            "has_internet_banking",
            "has_debit_card",
        ]
        diversity_count = 0
        for col in categorical_diversity_cols:
            if col in X_new.columns:
                diversity_count += (X_new[col] == "Have now").astype(int)
        X_new["banking_diversity_score"] = diversity_count
        X_new["is_banking_diverse"] = (diversity_count >= 3).astype(int)

        risk_cols = ["attitude_worried_shutdown", "current_problem_cash_flow", "problem_sourcing_money"]
        risk_score = 0
        for col in risk_cols:
            if col in X_new.columns:
                risk_score += (X_new[col] == "Yes").astype(int)
        X_new["comprehensive_risk_score"] = risk_score
        X_new["is_high_risk"] = (risk_score >= 2).astype(int)

        compliance_cols = ["compliance_income_tax", "keeps_financial_records"]
        compliance_score = 0
        for col in compliance_cols:
            if col in X_new.columns:
                compliance_score += (X_new[col] == "Yes").astype(int)
        X_new["compliance_responsibility_score"] = compliance_score
        X_new["is_highly_compliant"] = (compliance_score == 2).astype(int)

        return X_new

    def create_cluster_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create cluster feature using KMeans on numeric columns (fit/transform safe)."""
        X_new = X.copy()
        if KMeans is None:
            return X_new

        numeric_cols = [c for c in X_new.columns if pd.api.types.is_numeric_dtype(X_new[c]) and c != "cluster"]
        if not numeric_cols:
            return X_new

        if fit:
            self.cluster_numeric_cols = numeric_cols
            from sklearn.preprocessing import StandardScaler as _StandardScaler

            self.kmeans_scaler = _StandardScaler()

            X_scaled = self.kmeans_scaler.fit_transform(
                X_new[self.cluster_numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            )

            self.kmeans = KMeans(
                n_clusters=getattr(config, "CLUSTER_N_CLUSTERS", 4),
                random_state=getattr(config, "RANDOM_SEED", 42),
                n_init=getattr(config, "CLUSTER_N_INIT", 10),
            )
            X_new["cluster"] = self.kmeans.fit_predict(X_scaled)
        else:
            if not hasattr(self, "kmeans") or not hasattr(self, "kmeans_scaler") or not hasattr(self, "cluster_numeric_cols"):
                return X_new

            cols = [c for c in self.cluster_numeric_cols if c in X_new.columns]
            if not cols:
                return X_new

            X_scaled = self.kmeans_scaler.transform(
                X_new[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            )
            X_new["cluster"] = self.kmeans.predict(X_scaled)

        return X_new

    # --------------------------
    # Missing values / categorical encoding (fit/transform safe)
    # --------------------------
    def handle_missing_values(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values.
        FIX: store medians/modes on fit, reuse on transform.
        FIX: IterativeImputer transform uses transform(), not fit_transform().
        """
        X_processed = X.copy()

        num_cols = [c for c in getattr(config, "NUMERICAL_FEATURES", []) if c in X_processed.columns]
        cat_cols = [c for c in getattr(config, "CATEGORICAL_FEATURES", []) if c in X_processed.columns]

        use_iter = bool(getattr(config, "USE_ITERATIVE_IMPUTER", False)) and (IterativeImputer is not None)

        # first handle categoricals with mode (always)
        if fit:
            for col in cat_cols:
                mode_series = X_processed[col].mode(dropna=True)
                mode_val = mode_series.iloc[0] if len(mode_series) else "Unknown"
                self.categorical_modes_[col] = mode_val
                X_processed[col] = X_processed[col].fillna(mode_val)
        else:
            for col in cat_cols:
                fill_val = self.categorical_modes_.get(col, "Unknown")
                X_processed[col] = X_processed[col].fillna(fill_val)

        if use_iter and len(num_cols) > 0:
            if fit:
                params = getattr(config, "ITERATIVE_IMPUTER_PARAMS", {}) or {}
                imputer = IterativeImputer(**params, random_state=getattr(config, "RANDOM_SEED", 42))
                try:
                    imputed = imputer.fit_transform(X_processed[num_cols])
                    self.imputers["iterative"] = imputer
                    X_processed[num_cols] = imputed
                except Exception:
                    # fallback to medians
                    for col in num_cols:
                        s = pd.to_numeric(X_processed[col], errors="coerce")
                        med = float(s.median()) if np.isfinite(s.median()) else 0.0
                        self.numeric_medians_[col] = med
                        X_processed[col] = s.fillna(med)
            else:
                imputer = self.imputers.get("iterative")
                if imputer is not None:
                    try:
                        imputed = imputer.transform(X_processed[num_cols])
                        X_processed[num_cols] = imputed
                    except Exception:
                        for col in num_cols:
                            med = self.numeric_medians_.get(col, float(pd.to_numeric(X_processed[col], errors="coerce").median()))
                            X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce").fillna(med)
                else:
                    for col in num_cols:
                        med = self.numeric_medians_.get(col, float(pd.to_numeric(X_processed[col], errors="coerce").median()))
                        X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce").fillna(med)

            return X_processed

        # default simple imputation
        if fit:
            for col in num_cols:
                s = pd.to_numeric(X_processed[col], errors="coerce")
                med = float(s.median()) if np.isfinite(s.median()) else 0.0
                self.numeric_medians_[col] = med
                X_processed[col] = s.fillna(med)
        else:
            for col in num_cols:
                med = self.numeric_medians_.get(col, float(pd.to_numeric(X_processed[col], errors="coerce").median()))
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce").fillna(med)

        return X_processed

    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder (fit/transform safe)."""
        X_encoded = X.copy()

        categorical_cols = set(getattr(config, "CATEGORICAL_FEATURES", []))
        for col in X_encoded.columns:
            if X_encoded[col].dtype == "object" or X_encoded[col].dtype.name == "category":
                categorical_cols.add(col)

        for col in categorical_cols:
            if col not in X_encoded.columns:
                continue

            if fit:
                le = LabelEncoder()
                vals = X_encoded[col].astype(str)
                X_encoded[col] = le.fit_transform(vals)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    # if encoder missing, just factorize (last resort)
                    X_encoded[col] = pd.factorize(X_encoded[col].astype(str))[0]
                    continue

                vals = X_encoded[col].astype(str)
                known = set(le.classes_)
                fallback = le.classes_[0] if len(le.classes_) else ""
                vals = vals.apply(lambda x: x if x in known else fallback)
                X_encoded[col] = le.transform(vals)

        return X_encoded

    # --------------------------
    # Main entry points
    # --------------------------
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit preprocessing and transform training data."""
        X_processed = self.handle_missing_values(X, fit=True)

        # fit global stats on (imputed) raw features before feature creation
        self._fit_feature_stats(X_processed)

        if y is not None:
            self.fit_target_encoding(X_processed, y)
            # use OOF encodings for train if available
            if hasattr(self, "oof_target_encodings") and self.oof_target_encodings:
                for col, oof_df in self.oof_target_encodings.items():
                    X_processed = pd.concat([X_processed, oof_df], axis=1)
                    if col in X_processed.columns:
                        X_processed.drop(columns=[col], inplace=True)
            else:
                X_processed = self.apply_target_encoding(X_processed)

        # feature blocks (fit-safe)
        X_processed = self.create_financial_features(X_processed, fit=True)
        X_processed = self.create_age_features(X_processed, fit=True)
        X_processed = self.create_insurance_banking_features(X_processed)
        X_processed = self.create_attitude_features(X_processed)
        X_processed = self.create_interaction_features(X_processed, fit=True)
        X_processed = self.create_pairwise_numeric_interactions(X_processed)
        X_processed = self.create_groupby_agg_features(X_processed, fit=True)
        X_processed = self.create_statistical_features(X_processed, fit=True)
        X_processed = self.create_advanced_statistical_features(X_processed, fit=True)
        X_processed = self.create_distribution_features(X_processed)

        # encode remaining categoricals
        X_processed = self.encode_categorical_features(X_processed, fit=True)

        # final NA cleanup
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                X_processed[col] = X_processed[col].astype("category").cat.codes
            X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce").fillna(0)

        # clustering at end
        X_processed = self.create_cluster_features(X_processed, fit=True)

        self.fitted_features = list(X_processed.columns)
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessing."""
        X_processed = self.handle_missing_values(X, fit=False)

        # target encoding with full mapping
        X_processed = self.apply_target_encoding(X_processed)

        # feature blocks (transform-safe)
        X_processed = self.create_financial_features(X_processed, fit=False)
        X_processed = self.create_age_features(X_processed, fit=False)
        X_processed = self.create_insurance_banking_features(X_processed)
        X_processed = self.create_attitude_features(X_processed)
        X_processed = self.create_interaction_features(X_processed, fit=False)
        X_processed = self.create_pairwise_numeric_interactions(X_processed)
        X_processed = self.create_groupby_agg_features(X_processed, fit=False)
        X_processed = self.create_statistical_features(X_processed, fit=False)
        X_processed = self.create_advanced_statistical_features(X_processed, fit=False)
        X_processed = self.create_distribution_features(X_processed)

        X_processed = self.encode_categorical_features(X_processed, fit=False)

        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                X_processed[col] = X_processed[col].astype("category").cat.codes
            X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce").fillna(0)

        X_processed = self.create_cluster_features(X_processed, fit=False)

        # align to train features
        if self.fitted_features is None:
            self.fitted_features = list(X_processed.columns)

        missing_cols = set(self.fitted_features) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0

        X_processed = X_processed[self.fitted_features]
        return X_processed


def preprocess_data(X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Main preprocessing function used by training/inference scripts."""
    if fit:
        pipeline = PreprocessingPipeline()
        X_processed = pipeline.fit_transform(X, y)
        with open(config.PREPROCESSING_FILE, "wb") as f:
            pickle.dump(pipeline, f)
    else:
        pipeline = load_preprocessing_pipeline()
        X_processed = pipeline.transform(X)

    return X_processed, y


def save_preprocessing_pipeline(pipeline: PreprocessingPipeline) -> None:
    with open(config.PREPROCESSING_FILE, "wb") as f:
        pickle.dump(pipeline, f)


def load_preprocessing_pipeline() -> PreprocessingPipeline:
    with open(config.PREPROCESSING_FILE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_loader import load_train_data, load_test_data

    X_train, y_train = load_train_data()
    X_test, test_ids = load_test_data()

    print("Original data shape:", X_train.shape)

    X_train_processed, _ = preprocess_data(X_train, y_train, fit=True)
    print("Processed train data shape:", X_train_processed.shape)

    X_test_processed, _ = preprocess_data(X_test, fit=False)
    print("Processed test data shape:", X_test_processed.shape)

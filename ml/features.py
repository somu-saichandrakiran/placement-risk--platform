# ml/features.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class PlacementFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Academic trajectory
        X["academic_trajectory"]  = X["cgpa"] * 10 - X["twelfth_percent"]
        X["academic_consistency"] = X[["tenth_percent", "twelfth_percent"]].std(axis=1)

        # Backlog signals
        X["backlog_flag"]     = (X["active_backlogs"] > 0).astype(int)
        X["backlog_severity"] = np.log1p(X["active_backlogs"])

        # Experience
        X["experience_score"] = (
            X["internship_count"] * 2.0 +
            X["project_count"]    * 1.0 +
            X["hackathon_count"]  * 1.5
        )
        X["has_internship"] = (X["internship_count"] > 0).astype(int)

        # CGPA thresholds
        X["cgpa_above_7"] = (X["cgpa"] >= 7.0).astype(int)
        X["cgpa_above_8"] = (X["cgpa"] >= 8.0).astype(int)
        X["cgpa_below_6"] = (X["cgpa"] <  6.0).astype(int)

        # Interactions
        X["cgpa_x_internship"] = X["cgpa"] * X["internship_count"]
        X["recovered_backlog"]  = (
            X["backlog_flag"] * (X["cgpa"] >= 7.5).astype(int)
        )

        # College tier
        tier_map = {"T1": 3, "T2": 2, "T3": 1}
        if "college_tier" in X.columns:
            X["tier_rank"] = X["college_tier"].map(tier_map).fillna(1)

        return X


def load_and_clean(filepath: str):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows")

    rename_map = {
        "sl_no":          "student_id",
        "ssc_p":          "tenth_percent",
        "ssc_b":          "tenth_board",
        "hsc_p":          "twelfth_percent",
        "hsc_b":          "twelfth_board",
        "hsc_s":          "specialization_12th",
        "degree_p":       "cgpa",
        "degree_t":       "degree_type",
        "workex":         "work_experience",
        "etest_p":        "employability_score",
        "specialisation": "specialization",
        "mba_p":          "mba_percent",
        "status":         "placed",
        "salary":         "salary_inr",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Encode target
    df["placed"] = (df["placed"].str.strip().str.lower() == "placed").astype(int)

    # Simulate richer features
    rng = np.random.default_rng(42)
    df["active_backlogs"]  = rng.integers(0, 4,  size=len(df))
    df["internship_count"] = rng.integers(0, 4,  size=len(df))
    df["project_count"]    = rng.integers(0, 8,  size=len(df))
    df["hackathon_count"]  = rng.integers(0, 3,  size=len(df))
    df["college_tier"]     = rng.choice(["T1","T2","T3"], size=len(df), p=[0.2,0.5,0.3])

    # Work experience flag
    if "work_experience" in df.columns:
        df["work_exp_flag"] = (
            df["work_experience"].str.strip().str.lower() == "yes"
        ).astype(int)

    df = df.dropna(subset=["placed"])

    y_placement = df["placed"]
    y_salary    = df["salary_inr"].fillna(0)

    drop_cols = ["student_id","placed","salary_inr",
                 "tenth_board","twelfth_board","work_experience"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info(f"Features: {X.shape[1]}, Placement rate: {y_placement.mean():.1%}")
    return X, y_placement, y_salary
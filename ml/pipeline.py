# ml/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from ml.features import PlacementFeatureEngineer

NUMERIC_FEATURES = [
    "tenth_percent", "twelfth_percent", "cgpa",
    "employability_score", "mba_percent",
    "internship_count", "project_count", "hackathon_count",
    "active_backlogs", "work_exp_flag",
    # engineered
    "academic_trajectory", "academic_consistency",
    "backlog_severity", "experience_score",
    "cgpa_x_internship", "tier_rank",
]

CATEGORICAL_FEATURES = [
    "specialization_12th", "degree_type", "specialization",
]

BINARY_FEATURES = [
    "backlog_flag", "has_internship",
    "cgpa_above_7", "cgpa_above_8", "cgpa_below_6",
    "recovered_backlog",
]


def build_placement_pipeline() -> CalibratedClassifierCV:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", "passthrough",        BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    base_pipeline = Pipeline([
        ("engineer",     PlacementFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier",   xgb),
    ])

    calibrated = CalibratedClassifierCV(
        base_pipeline,
        cv=5,
        method="isotonic",
    )
    return calibrated
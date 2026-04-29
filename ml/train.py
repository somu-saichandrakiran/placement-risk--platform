# ml/train.py
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, confusion_matrix, precision_recall_curve
from sklearn.dummy import DummyClassifier
from mlflow.models import infer_signature
from loguru import logger

from ml.features import load_and_clean
from ml.pipeline import build_placement_pipeline


def find_best_threshold(y_true, y_prob) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[min(best_idx, len(thresholds) - 1)])


def train(data_path: str = "data/raw/placements_raw.csv"):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("placement-risk")

    # Load data
    X, y, y_salary = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    with mlflow.start_run(run_name=f"xgb_{datetime.now():%Y%m%d_%H%M}"):

        # Dummy baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_auc = roc_auc_score(
            y_test, dummy.predict_proba(X_test)[:, 1]
        )
        mlflow.log_metric("dummy_roc_auc", round(dummy_auc, 4))

        # Cross validation
        pipeline = build_placement_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["roc_auc", "f1_macro", "neg_brier_score"],
            return_train_score=True,
        )

        cv_metrics = {
            "cv_roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
            "cv_roc_auc_std":  round(cv_results["test_roc_auc"].std(), 4),
            "cv_f1_mean":      round(cv_results["test_f1_macro"].mean(), 4),
            "cv_brier_mean":   round(-cv_results["test_neg_brier_score"].mean(), 4),
            "cv_overfit_gap":  round(
                cv_results["train_roc_auc"].mean() -
                cv_results["test_roc_auc"].mean(), 4
            ),
        }

        # Final fit
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        best_threshold = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        test_metrics = {
            "test_roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "test_f1_placed": round(f1_score(y_test, y_pred), 4),
            "test_f1_macro":  round(f1_score(y_test, y_pred, average="macro"), 4),
            "test_brier":     round(brier_score_loss(y_test, y_prob), 4),
            "best_threshold": round(best_threshold, 4),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]),
            "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        }

        # Gates
        gates = {
            "roc_auc_gate": test_metrics["test_roc_auc"]  >= 0.72,
            "f1_gate":      test_metrics["test_f1_macro"]  >= 0.60,
            "brier_gate":   test_metrics["test_brier"]     <= 0.22,
            "beats_dummy":  test_metrics["test_roc_auc"]   > dummy_auc + 0.05,
        }

        all_metrics = {
            **cv_metrics, **test_metrics,
            "train_rows":     len(X_train),
            "test_rows":      len(X_test),
            "placement_rate": round(float(y.mean()), 4),
            **{f"gate_{k}": int(v) for k, v in gates.items()},
        }

        mlflow.log_params({
            "model_type":  "XGBoost+CalibratedCV",
            "cv_folds":    5,
            "calibration": "isotonic",
            "threshold":   round(best_threshold, 4),
        })
        mlflow.log_metrics(all_metrics)

        # Register model
        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="placement_model",
            signature=signature,
            registered_model_name="placement-risk-classifier",
            pip_requirements=[
                "scikit-learn",
                "xgboost",
                "numpy",
                "pandas",
                "shap",
            ],
        )

        run_id = mlflow.active_run().info.run_id

        # Print summary
        print("\n" + "="*50)
        print("  TRAINING COMPLETE")
        print("="*50)
        print(f"  Dummy ROC-AUC : {dummy_auc:.4f}")
        print(f"  CV  ROC-AUC   : {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
        print(f"  Test ROC-AUC  : {test_metrics['test_roc_auc']:.4f}")
        print(f"  Test F1 macro : {test_metrics['test_f1_macro']:.4f}")
        print(f"  Test Brier    : {test_metrics['test_brier']:.4f}")
        print(f"  Threshold     : {best_threshold:.4f}")
        print(f"  Overfit gap   : {cv_metrics['cv_overfit_gap']:.4f}")
        print("-"*50)
        for gate, passed in gates.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {gate}")
        print("="*50)
        print(f"  run_id: {run_id}")
        print(f"  MLflow: http://localhost:5000\n")

        return pipeline, run_id, all_metrics


if __name__ == "__main__":
    train()
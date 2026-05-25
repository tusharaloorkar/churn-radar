# =============================================================================
# src/train.py - Model Training Script (Project 2: + MLflow Tracking)
# =============================================================================
#
# WHAT'S NEW IN PROJECT 2?
#   MLflow experiment tracking is added throughout this script.
#   Every time you run "python src/train.py", MLflow automatically records:
#     - Parameters: n_estimators, max_depth, learning_rate, etc.
#     - Metrics: accuracy, F1 score, ROC-AUC for BOTH models
#     - Artifacts: the saved model file (.pkl)
#     - Tags: which model won, dataset size, Python version
#
#   After training, run: mlflow ui
#   Then open: http://localhost:5000
#   You'll see a table comparing every run you've ever done.
#
# WHAT IS MLFLOW?
#   MLflow is an open-source platform for managing the ML lifecycle.
#   It solves a very real problem: after 10 training runs with different
#   settings, which one was best? Without MLflow, you'd rely on memory
#   or scattered notes. MLflow logs everything automatically.
#
#   Core concepts:
#     - Experiment: a named group of related runs (e.g. "churn-radar")
#     - Run: one execution of train.py — has its own params + metrics
#     - Artifact: any file saved to a run (model, plots, reports)
#     - Registry: a central store of "official" model versions
#
# HOW TO RUN:
#   python src/train.py           <- trains + logs to MLflow
#   mlflow ui                     <- open the tracking dashboard
# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_full_pipeline

# --- Configuration ---
RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_SAVE_PATH = "model/best_model.pkl"
METRICS_SAVE_PATH = "model/metrics.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- MLflow Configuration ---
# EXPERIMENT_NAME groups all your training runs together.
# Think of it like a folder: all runs of "churn-radar" sit inside it.
EXPERIMENT_NAME = "churn-radar"

# Where MLflow stores its data. "mlruns/" is a local folder it creates
# automatically. In production teams use a central server instead.
MLFLOW_TRACKING_URI = "mlruns"


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Evaluate a trained model on the test set and return metrics.

    METRICS EXPLAINED:
      Accuracy  = % of all predictions that were correct
                  (misleading on imbalanced data — a model predicting
                  "no churn" always gets 73.5% accuracy but is useless)

      F1 Score  = harmonic mean of Precision and Recall
                  Precision = of predicted churners, how many actually churned?
                  Recall    = of actual churners, how many did we catch?
                  F1 balances both — 1.0 is perfect, 0.0 is worst

      ROC-AUC   = how well the model ranks churners above non-churners
                  across ALL thresholds. 1.0 = perfect, 0.5 = random guessing
                  This is our primary metric for model selection.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4), "roc_auc": round(roc_auc, 4)}


def train_random_forest(X_train, y_train) -> tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest and log it as an MLflow nested run.

    WHAT IS A NESTED RUN?
      Our main MLflow run covers the entire training session.
      Inside it, we create one nested run per model so each model's
      params and metrics are stored separately but linked to the parent.

      Structure:
        Run: "churn-radar training session"
          - Nested Run: "RandomForest"  <- params + metrics for RF
          - Nested Run: "XGBoost"       <- params + metrics for XGB

    RANDOM FOREST HYPERPARAMETERS:
      n_estimators=200    : build 200 decision trees, vote for final answer
      max_depth=10        : limit tree depth to prevent memorizing training data
      class_weight=balanced: upweight churners so the model doesn't ignore them
    """
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }

    print("\n[train] Training Random Forest...")

    with mlflow.start_run(run_name="RandomForest", nested=True):
        # Log every hyperparameter so you can reproduce this exact model later
        # mlflow.log_params() saves a dict of name:value pairs
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_global)
        y_prob = model.predict_proba(X_test_global)[:, 1]
        metrics = {
            "accuracy": round(accuracy_score(y_test_global, y_pred), 4),
            "f1": round(f1_score(y_test_global, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test_global, y_prob), 4),
        }

        # Log all metrics — these appear as columns in the MLflow UI table
        mlflow.log_metrics(metrics)

        # Log feature importances as individual metrics so you can plot them
        for feat, imp in zip(X_train.columns, model.feature_importances_):
            mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))

        # Log the model as an MLflow artifact (queryable, loadable, versioned)
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

        print("[train] Random Forest training complete.")
        print(f"[mlflow] Logged RandomForest run: accuracy={metrics['accuracy']}, roc_auc={metrics['roc_auc']}")

    return model, metrics


def train_xgboost(X_train, y_train) -> tuple[XGBClassifier, dict]:
    """
    Train an XGBoost model and log it as an MLflow nested run.

    XGBOOST HYPERPARAMETERS:
      n_estimators=200  : number of sequential boosting rounds
      max_depth=6       : tree depth (lower = less overfit, XGB is sensitive here)
      learning_rate=0.1 : step size per round. Small = more rounds needed but better fit
      subsample=0.8     : use 80% of rows per tree (adds randomness, prevents overfit)
      colsample_bytree=0.8: use 80% of features per tree (same purpose as subsample)
      scale_pos_weight  : ratio of negatives/positives — corrects class imbalance
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = round(float(neg_count / pos_count), 4)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    }

    print("\n[train] Training XGBoost...")
    print(f"[train] Class ratio (neg/pos) = {scale} -> scale_pos_weight={scale}")

    with mlflow.start_run(run_name="XGBoost", nested=True):
        mlflow.log_params(params)

        model = XGBClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test_global)
        y_prob = model.predict_proba(X_test_global)[:, 1]
        metrics = {
            "accuracy": round(accuracy_score(y_test_global, y_pred), 4),
            "f1": round(f1_score(y_test_global, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test_global, y_prob), 4),
        }

        mlflow.log_metrics(metrics)

        for feat, imp in zip(X_train.columns, model.feature_importances_):
            mlflow.log_metric(f"importance_{feat}", round(float(imp), 4))

        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

        print("[train] XGBoost training complete.")
        print(f"[mlflow] Logged XGBoost run: accuracy={metrics['accuracy']}, roc_auc={metrics['roc_auc']}")

    return model, metrics


def print_feature_importances(model, feature_names: list, model_name: str):
    """Print the top features the model used to make predictions."""
    if hasattr(model, "feature_importances_"):
        pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
        print(f"\n[train] {model_name} - Top Feature Importances:")
        for feat, imp in pairs[:8]:
            bar = "#" * int(imp * 50)
            print(f"  {feat:<25} {imp:.4f}  {bar}")


# Module-level globals used by nested run functions
# (MLflow nested runs can't easily pass data through return values)
X_test_global = None
y_test_global = None


def main():
    global X_test_global, y_test_global

    print("=" * 60)
    print("  CHURN RADAR - MODEL TRAINING + MLFLOW TRACKING")
    print("=" * 60)

    # --- Setup MLflow ---
    # set_tracking_uri tells MLflow where to save data.
    # "mlruns" = a local folder in your project directory.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # set_experiment creates the experiment if it doesn't exist,
    # or finds the existing one if it does.
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n[mlflow] Experiment: '{EXPERIMENT_NAME}'")
    print(f"[mlflow] Tracking URI: {os.path.abspath(MLFLOW_TRACKING_URI)}")
    print("[mlflow] After training, run: mlflow ui")
    print("[mlflow] Then open: http://localhost:5000\n")

    # --- Step 1: Load and preprocess data ---
    print("[train] Step 1: Loading and preprocessing data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {RAW_DATA_PATH}")
        print("Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        sys.exit(1)

    X, y = run_full_pipeline(RAW_DATA_PATH)

    # --- Step 2: Train/test split ---
    print(f"\n[train] Step 2: Splitting data ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Make test data available to nested run functions
    X_test_global = X_test
    y_test_global = y_test
    print(f"[train] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Step 3 & 4: Train + evaluate inside the PARENT MLflow run ---
    # Everything inside this "with" block belongs to one parent run.
    # start_run() creates a new row in the MLflow experiment table.
    with mlflow.start_run(run_name="churn-radar-training") as parent_run:
        run_id = parent_run.info.run_id
        print(f"\n[mlflow] Started parent run: {run_id}")

        # Log dataset info as tags (descriptive labels, not numeric metrics)
        mlflow.set_tags({
            "dataset": "Telco Customer Churn",
            "dataset_rows": len(X),
            "features": len(X.columns),
            "churn_rate": f"{y.mean():.2%}",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "python_version": sys.version.split()[0],
        })

        # Log the feature list as a JSON artifact
        # Artifacts are files attached to a run — useful for documentation
        os.makedirs("model", exist_ok=True)
        features_path = "model/features.json"
        with open(features_path, "w") as f:
            json.dump(list(X.columns), f, indent=2)
        mlflow.log_artifact(features_path)

        # --- Train both models (each as a nested run) ---
        print("\n[train] Step 3: Training models...")
        rf_model, rf_metrics = train_random_forest(X_train, y_train)
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train)

        # --- Full evaluation printout ---
        print("\n[train] Step 4: Full evaluation on test set...")
        evaluate_model(rf_model, X_test, y_test, "Random Forest")
        evaluate_model(xgb_model, X_test, y_test, "XGBoost")

        # --- Pick the best model ---
        print("\n[train] Step 5: Selecting best model...")
        if xgb_metrics["roc_auc"] >= rf_metrics["roc_auc"]:
            best_model = xgb_model
            best_name = "XGBoost"
            best_metrics = xgb_metrics
        else:
            best_model = rf_model
            best_name = "Random Forest"
            best_metrics = rf_metrics

        print(f"\n  Winner: {best_name} (ROC-AUC: {best_metrics['roc_auc']})")
        print_feature_importances(best_model, list(X.columns), best_name)

        # Log the winner's metrics on the PARENT run so they appear
        # at the top level in the MLflow UI — easy to compare across sessions
        mlflow.log_metrics({
            f"best_{k}": v for k, v in best_metrics.items()
        })
        mlflow.set_tag("best_model", best_name)

        # --- Save model + metrics ---
        print(f"\n[train] Step 6: Saving best model to {MODEL_SAVE_PATH}...")
        joblib.dump(best_model, MODEL_SAVE_PATH)
        mlflow.log_artifact(MODEL_SAVE_PATH)

        metrics_record = {
            "model_name": best_name,
            "run_id": run_id,
            "features": list(X.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": best_metrics,
            "all_models": {
                "RandomForest": rf_metrics,
                "XGBoost": xgb_metrics,
            },
        }
        with open(METRICS_SAVE_PATH, "w") as f:
            json.dump(metrics_record, f, indent=2)
        mlflow.log_artifact(METRICS_SAVE_PATH)

        print(f"[train] Metrics saved to {METRICS_SAVE_PATH}")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Model    : {best_name}")
    print(f"  Accuracy : {best_metrics['accuracy']*100:.1f}%")
    print(f"  F1 Score : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {best_metrics['roc_auc']:.4f}")
    print(f"  MLflow Run ID: {run_id}")
    print("=" * 60)
    print("\n  To view the MLflow dashboard:")
    print("  1. Run:  mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()

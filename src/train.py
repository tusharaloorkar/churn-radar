# =============================================================================
# src/train.py - Model Training Script
# =============================================================================
#
# WHAT DOES THIS SCRIPT DO?
#   This is the brain of the project. It:
#     1. Loads and preprocesses the raw dataset
#     2. Splits it into training data and test data
#     3. Trains two models (Random Forest and XGBoost)
#     4. Evaluates both models and picks the best one
#     5. Saves the best model to disk so the API can load it later
#
# HOW TO RUN:
#   From the churn-predictor/ folder:
#     python src/train.py
#
# WHAT IS A TRAINED MODEL?
#   A model is a mathematical function that maps input features → prediction.
#   "Training" means feeding the model thousands of examples (rows) so it
#   learns patterns: e.g., "customers on month-to-month contracts with high
#   monthly charges tend to churn more often."
#
# WHAT IS A .pkl FILE?
#   After training, we serialize (save) the model object using joblib.
#   joblib.dump(model, "model/best_model.pkl") writes the model to a binary
#   file. joblib.load("model/best_model.pkl") reads it back later.
#   This way the API doesn't need to retrain - it just loads the saved model.
# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

# Add the project root to Python's path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_full_pipeline

# --- Configuration ---
# These paths are relative to where you run the script (churn-predictor/)
RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_SAVE_PATH = "model/best_model.pkl"
METRICS_SAVE_PATH = "model/metrics.json"

# How much of the data to hold out for testing (20% here).
# The model NEVER sees test data during training - it's used only to
# measure how well the model generalizes to new, unseen customers.
TEST_SIZE = 0.2

# random_state=42 is a convention (the number doesn't matter, but fixing it
# means you get the same train/test split every time you run the script).
RANDOM_STATE = 42


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Run the trained model on the test set and compute evaluation metrics.

    KEY METRICS EXPLAINED:
      - Accuracy: % of all predictions that were correct.
          Simple but misleading when classes are imbalanced (e.g., 85% No-churn,
          15% churn - a model that always predicts "No" gets 85% accuracy!).

      - F1 Score: balances Precision and Recall. Better metric for imbalanced data.
          Precision = of all "churn" predictions, how many were right?
          Recall    = of all actual churners, how many did we catch?
          F1        = harmonic mean of the two. 1.0 = perfect, 0.0 = worst.

      - ROC-AUC: measures how well the model separates churners from non-churners
          across all possible decision thresholds. 1.0 = perfect, 0.5 = random.

    Args:
        model: trained sklearn/xgboost model
        X_test: test features
        y_test: true labels (0 or 1)
        model_name: just for printing

    Returns:
        metrics dict with accuracy, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (churn)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  (rows=Actual, cols=Predicted)")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4), "roc_auc": round(roc_auc, 4)}


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    WHAT IS RANDOM FOREST?
      A Random Forest builds many Decision Trees, each trained on a random
      subset of the data. The final prediction is a vote across all trees.
      "Forest" = many trees. "Random" = each tree sees random data + features.

      Advantages:
        - Works well out-of-the-box with minimal tuning
        - Handles mixed data (numbers + categories)
        - Gives feature importances (tells you which columns matter most)
        - Less likely to overfit than a single decision tree

    KEY HYPERPARAMETERS:
      n_estimators=200: build 200 trees (more = more accurate but slower)
      max_depth=10: limit tree depth to prevent memorizing training data
                    (overfitting)
      class_weight="balanced": automatically give more weight to the minority
                    class (churners). Without this, the model ignores rare
                    events and just predicts "No churn" for everything.
    """
    print("\n[train] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("[train] Random Forest training complete.")
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    WHAT IS XGBOOST?
      XGBoost (eXtreme Gradient Boosting) builds trees sequentially - each
      new tree learns from the mistakes of the previous one. It's called
      "boosting" because each step boosts the model's overall performance.

      Compared to Random Forest:
        - Usually achieves higher accuracy on tabular data
        - Trains faster (optimized C++ backend)
        - More hyperparameters to tune, but great defaults exist

    KEY HYPERPARAMETERS:
      n_estimators=200: number of boosting rounds
      max_depth=6: tree depth (XGBoost is sensitive to this - keep it low)
      learning_rate=0.1: how much each tree corrects the previous one.
                        Small = slow learning but more accurate. Too large = overfit.
      scale_pos_weight: handles class imbalance. Set to ratio of negatives/positives.
    """
    print("\n[train] Training XGBoost...")

    # Calculate class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count
    print(f"[train] Class ratio (neg/pos) = {scale:.2f} -> setting scale_pos_weight={scale:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,           # use 80% of rows per tree (reduces overfitting)
        colsample_bytree=0.8,    # use 80% of features per tree
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[train] XGBoost training complete.")
    return model


def print_feature_importances(model, feature_names: list, model_name: str):
    """
    Print which features the model found most useful for predicting churn.

    WHY THIS MATTERS:
      Feature importances tell you *why* the model predicts what it predicts.
      If "Contract" is the top feature, it means contract type is the biggest
      driver of churn - useful business insight, not just a number on a screen.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print(f"\n[train] {model_name} - Top Feature Importances:")
        for feat, imp in pairs[:8]:  # top 8
            bar = "#" * int(imp * 50)
            print(f"  {feat:<25} {imp:.4f}  {bar}")


def main():
    """
    Main training function - runs the entire training pipeline end to end.
    """
    print("=" * 60)
    print("  CHURN PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    # --- Step 1: Load and preprocess data ---
    print("\n[train] Step 1: Loading and preprocessing data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {RAW_DATA_PATH}")
        print("Please download the Telco Customer Churn dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"and place the CSV file at: {RAW_DATA_PATH}")
        sys.exit(1)

    X, y = run_full_pipeline(RAW_DATA_PATH)

    # --- Step 2: Train/test split ---
    print(f"\n[train] Step 2: Splitting data ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        # stratify=y ensures the churn ratio is preserved in both splits.
        # Without this, by chance you might get a test set with very few churners.
    )
    print(f"[train] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Step 3: Train both models ---
    print("\n[train] Step 3: Training models...")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # --- Step 4: Evaluate both models ---
    print("\n[train] Step 4: Evaluating models on test set...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # --- Step 5: Pick the best model based on ROC-AUC ---
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

    # Print feature importances for the winner
    print_feature_importances(best_model, list(X.columns), best_name)

    # --- Step 6: Save model and metrics ---
    print(f"\n[train] Step 6: Saving model to {MODEL_SAVE_PATH}...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH)

    # Save metrics as JSON for reference and future experiment tracking
    metrics_record = {
        "model_name": best_name,
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

    print(f"[train] Metrics saved to {METRICS_SAVE_PATH}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Model: {best_name}")
    print(f"  Accuracy : {best_metrics['accuracy']*100:.1f}%")
    print(f"  F1 Score : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {best_metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # This block only runs when you execute "python src/train.py" directly.
    # It does NOT run when another file imports from this module.
    main()

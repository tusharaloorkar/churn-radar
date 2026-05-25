# =============================================================================
# src/predict.py — Inference Logic
# =============================================================================
#
# WHAT IS "INFERENCE"?
#   Training = teaching the model using historical data (done once, offline)
#   Inference = using the trained model to make predictions on new data
#               (done many times, in real-time, via the API)
#
# WHY A SEPARATE FILE FROM train.py?
#   - train.py runs once, takes minutes, uses the full dataset
#   - predict.py runs in milliseconds, handles one customer at a time
#   - Keeping them separate makes each easier to understand and test
#
# THE CRITICAL RULE:
#   The preprocessing in predict.py MUST be IDENTICAL to train.py.
#   If you trained with StandardScaler, you MUST use the SAME scaler
#   (loaded from disk) when predicting. Using a new scaler would produce
#   completely different numbers and garbage predictions.
#   This is why we saved encoders.pkl and scaler.pkl during training.
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import encode_categorical_features, scale_numeric_features, FEATURE_COLUMNS

# Paths to the saved model and preprocessing artifacts
MODEL_PATH = "model/best_model.pkl"
ENCODER_PATH = "model/encoders.pkl"
SCALER_PATH = "model/scaler.pkl"


class ChurnPredictor:
    """
    A class that wraps the trained model and handles end-to-end prediction.

    WHY A CLASS?
      When the FastAPI server starts, we load the model ONCE into memory.
      Every subsequent prediction call reuses the already-loaded model.
      Loading from disk is slow (~100ms). Predicting with a loaded model
      is fast (~1ms). Using a class lets us keep the model "alive" in memory.

    USAGE:
      predictor = ChurnPredictor()          # loads model (once at startup)
      result = predictor.predict({...})     # fast prediction (each request)
    """

    def __init__(self):
        """
        Load the trained model and preprocessing artifacts from disk.
        This runs once when the FastAPI app starts up.
        """
        print("[predict] Loading model and preprocessing artifacts...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run 'python src/train.py' first to train the model."
            )

        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        print(f"[predict] Model loaded: {type(self.model).__name__}")
        print("[predict] Ready to make predictions.")

    def preprocess_input(self, customer_data: dict) -> pd.DataFrame:
        """
        Transform a single customer's raw data into the format the model expects.

        This mirrors the preprocessing in preprocess.py, but:
          - Works on a single row (not the full dataset)
          - Uses fit=False (loads saved encoders/scalers instead of relearning)

        Args:
            customer_data: dict with customer fields, e.g.:
              {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1572.0,
                "Contract": "Month-to-month",
                ...
              }

        Returns:
            X: a 1-row DataFrame ready for model.predict()
        """
        # Convert the dict to a single-row DataFrame
        # pd.DataFrame([dict]) creates a DataFrame with one row
        df = pd.DataFrame([customer_data])

        # Apply binary encoding (Yes/No → 1/0)
        binary_columns = [
            "Partner", "Dependents", "PhoneService", "PaperlessBilling",
            "MultipleLines", "OnlineSecurity", "TechSupport",
        ]
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Female": 1, "Male": 0}).fillna(0).astype(int)

        # Apply multi-class encoding using saved LabelEncoders
        multi_class_columns = ["Contract", "PaymentMethod", "InternetService"]
        for col in multi_class_columns:
            if col in df.columns and col in self.encoders:
                le = self.encoders[col]
                df[col] = le.transform(df[col].astype(str))

        # Apply scaling using the saved StandardScaler
        numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        df[existing_numeric] = self.scaler.transform(df[existing_numeric])

        # Select only the feature columns in the right order
        available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        X = df[available_features]

        return X

    def predict(self, customer_data: dict) -> dict:
        """
        Make a churn prediction for a single customer.

        Args:
            customer_data: raw customer data as a dict (matching API input schema)

        Returns:
            A dict with:
              - churn: bool — True if model predicts the customer will churn
              - churn_probability: float — confidence score (0.0 to 1.0)
              - risk_level: str — "Low", "Medium", or "High" based on probability
              - message: human-readable interpretation
        """
        # Step 1: Preprocess the raw input
        X = self.preprocess_input(customer_data)

        # Step 2: Get prediction (0 or 1) and probability
        # predict() returns an array like [1] → we take [0] for the scalar value
        prediction = int(self.model.predict(X)[0])

        # predict_proba() returns [[prob_class0, prob_class1]]
        # [:, 1] gives us the probability of churn (class 1)
        churn_probability = float(self.model.predict_proba(X)[0][1])

        # Step 3: Interpret the probability as a risk level
        if churn_probability >= 0.7:
            risk_level = "High"
            message = "This customer has a high likelihood of churning. Immediate retention action recommended."
        elif churn_probability >= 0.4:
            risk_level = "Medium"
            message = "This customer shows moderate churn risk. Consider proactive outreach."
        else:
            risk_level = "Low"
            message = "This customer is likely to stay. No immediate action needed."

        return {
            "churn": bool(prediction),
            "churn_probability": round(churn_probability, 4),
            "risk_level": risk_level,
            "message": message,
        }


def get_predictor() -> ChurnPredictor:
    """
    Factory function used by FastAPI's dependency injection system.

    WHAT IS DEPENDENCY INJECTION?
      Instead of creating a new ChurnPredictor() on every API request
      (which would reload the model from disk each time — very slow),
      FastAPI calls this function once and reuses the same predictor object
      for all requests. This pattern is called "dependency injection".

    Returns:
        A singleton ChurnPredictor instance
    """
    return _predictor_instance


# Module-level singleton — created when the module is first imported
# (i.e., when the FastAPI app starts). All API requests share this instance.
_predictor_instance = None


def initialize_predictor():
    """
    Called once at API startup to load the model into memory.
    Sets the module-level _predictor_instance.
    """
    global _predictor_instance
    _predictor_instance = ChurnPredictor()

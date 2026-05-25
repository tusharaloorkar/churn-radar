# =============================================================================
# src/preprocess.py — Data Cleaning and Feature Engineering
# =============================================================================
#
# WHAT IS PREPROCESSING?
#   Raw data from the real world is messy. Before a machine learning model
#   can learn from it, we need to:
#     1. Fix or remove bad/missing values
#     2. Convert text categories into numbers (ML models only understand numbers)
#     3. Scale numbers so large values don't dominate small ones
#
# WHY A SEPARATE FILE?
#   We could do all this inside train.py, but separating it means:
#     - We can reuse the same logic in predict.py (crucial! the API must
#       preprocess incoming data exactly the same way as training data)
#     - Easier to read, test, and debug each step in isolation
#
# ABOUT THE DATASET (Telco Customer Churn):
#   Each row = one customer. Columns include:
#     - tenure: how many months they've been a customer
#     - MonthlyCharges: what they pay per month
#     - TotalCharges: total amount paid (sometimes blank — a data quality issue)
#     - Contract: "Month-to-month", "One year", or "Two year"
#     - Churn: "Yes" or "No" — this is what we're predicting (the TARGET)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


# This is the list of columns we'll actually use to train the model.
# We drop columns like 'customerID' because it's just an ID — it has
# no relationship with whether a customer churns.
FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "SeniorCitizen",
    "Dependents",
    "Partner",
    "MultipleLines",
    "PhoneService",
]

# The column we want to predict
TARGET_COLUMN = "Churn"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.

    A DataFrame is like a table (rows and columns). pandas reads the CSV
    and gives us a DataFrame we can manipulate with Python code.

    Args:
        filepath: path to the CSV file (e.g., "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    Returns:
        df: the raw DataFrame, exactly as it appears in the CSV
    """
    print(f"[preprocess] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[preprocess] Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data quality issues in the raw dataset.

    Issues we fix here:
      1. TotalCharges has spaces (" ") instead of 0 for brand-new customers
         (they haven't been charged yet). We convert those spaces to 0.
      2. After fixing spaces, TotalCharges is still a string column — we
         convert it to a proper number (float).

    Args:
        df: the raw DataFrame from load_data()

    Returns:
        df: the cleaned DataFrame
    """
    print("[preprocess] Cleaning data...")

    # Step 1: Replace blank spaces in TotalCharges with 0
    # strip() removes leading/trailing whitespace from each value.
    # If after stripping it's an empty string "", we replace with "0".
    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", "0")

    # Step 2: Convert TotalCharges from string to float
    # "11.65" (string) → 11.65 (float). errors="coerce" turns any
    # remaining bad values into NaN (Not a Number) instead of crashing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Step 3: Drop any rows where TotalCharges is still NaN after the above
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    after = len(df)
    if before != after:
        print(f"[preprocess] Dropped {before - after} rows with null TotalCharges")

    # Step 4: Convert the target column "Churn" from "Yes"/"No" to 1/0
    # ML models need numbers. "Yes" → 1 (churned), "No" → 0 (stayed).
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    print(f"[preprocess] Clean data shape: {df.shape}")
    return df


def encode_categorical_features(df: pd.DataFrame, fit: bool = True, encoder_path: str = "model/encoders.pkl") -> pd.DataFrame:
    """
    Convert text columns (categorical features) into numbers.

    Machine learning models can't work with text like "Month-to-month"
    or "Electronic check". We convert them to numbers using two strategies:

    Strategy A — Binary columns (only 2 unique values):
      "Yes"/"No" → 1/0   (simple map)
      "Male"/"Female" → 1/0

    Strategy B — Multi-class columns (3+ unique values):
      LabelEncoder assigns a unique integer to each category.
      Example: "Month-to-month"→0, "One year"→1, "Two year"→2

    Why separate fit vs. transform?
      During TRAINING (fit=True): we *learn* what the categories are and
      save that mapping to disk. Example: learn that Contract has 3 values.
      During INFERENCE (fit=False): we *load* the saved mapping and apply
      it to new data. This ensures new data is encoded exactly the same way.
      If we re-fit on new data, the numbers might be assigned differently!

    Args:
        df: cleaned DataFrame
        fit: True when called from train.py, False when called from predict.py
        encoder_path: where to save/load the encoder mappings

    Returns:
        df: DataFrame with all text columns converted to numbers
    """
    print(f"[preprocess] Encoding categorical features (fit={fit})...")

    # --- Binary columns: simple Yes/No or similar 2-value columns ---
    binary_columns = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    for col in binary_columns:
        if col in df.columns:
            # Map "Yes" → 1, "No" → 0. Other values (like "No internet service")
            # become NaN, which we then fill with 0.
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # "gender" column: "Female" → 1, "Male" → 0
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 1, "Male": 0}).fillna(0).astype(int)

    # --- Multi-class columns: 3 or more unique text values ---
    multi_class_columns = ["Contract", "PaymentMethod", "InternetService"]

    if fit:
        # TRAINING: learn and save the encodings
        encoders = {}
        for col in multi_class_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                print(f"  {col} classes: {list(le.classes_)}")

        # Save encoders to disk so predict.py can load them later
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(encoders, encoder_path)
        print(f"[preprocess] Encoders saved to {encoder_path}")
    else:
        # INFERENCE: load the saved encodings and apply them
        encoders = joblib.load(encoder_path)
        for col in multi_class_columns:
            if col in df.columns and col in encoders:
                le = encoders[col]
                # transform() applies the learned mapping to new data
                df[col] = le.transform(df[col].astype(str))

    return df


def scale_numeric_features(df: pd.DataFrame, fit: bool = True, scaler_path: str = "model/scaler.pkl") -> pd.DataFrame:
    """
    Scale numeric features to a standard range.

    WHY SCALE?
      Imagine tenure ranges from 0–72 and MonthlyCharges from 20–120.
      Many ML algorithms are sensitive to the *magnitude* of numbers —
      larger values get treated as more important. Scaling puts all
      numeric features on the same footing.

    WHAT IS STANDARDSCALER?
      StandardScaler transforms each value using:
        scaled_value = (value - mean) / standard_deviation
      After scaling, each feature has mean=0 and std=1.
      This is called "Z-score normalization".

    Same fit/transform logic as encode_categorical_features above.

    Args:
        df: DataFrame with encoded categories
        fit: True during training, False during inference
        scaler_path: where to save/load the scaler

    Returns:
        df: DataFrame with scaled numeric columns
    """
    print(f"[preprocess] Scaling numeric features (fit={fit})...")

    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    existing_numeric = [col for col in numeric_columns if col in df.columns]

    if fit:
        scaler = StandardScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[preprocess] Scaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[existing_numeric] = scaler.transform(df[existing_numeric])

    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split the DataFrame into:
      - X: the input features (everything the model uses to make predictions)
      - y: the target label (what we're trying to predict: 0=stayed, 1=churned)

    This is the standard ML convention: X is the "features matrix",
    y is the "target vector".

    Args:
        df: fully preprocessed DataFrame

    Returns:
        X: pandas DataFrame of feature columns
        y: pandas Series of 0/1 churn labels
    """
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[available_features]
    y = df[TARGET_COLUMN]
    print(f"[preprocess] Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"[preprocess] Churn rate: {y.mean():.2%} ({y.sum()} churned out of {len(y)})")
    return X, y


def run_full_pipeline(filepath: str):
    """
    Convenience function that runs ALL preprocessing steps in order.
    Called by train.py to get training-ready data in one line.

    Pipeline order:
      load → clean → encode → scale → split X/y

    Args:
        filepath: path to raw CSV

    Returns:
        X, y: features and labels ready for model training
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_categorical_features(df, fit=True)
    df = scale_numeric_features(df, fit=True)
    X, y = get_features_and_target(df)
    return X, y

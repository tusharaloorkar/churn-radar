# =============================================================================
# app/main.py — FastAPI Application
# =============================================================================
#
# WHAT IS FASTAPI?
#   FastAPI is a Python web framework for building APIs (Application
#   Programming Interfaces). An API is a way for software to talk to other
#   software over HTTP — the same protocol your browser uses to load websites.
#
#   In our case:
#     - A client (curl, Postman, a website) sends a POST request with
#       customer data as JSON
#     - Our API receives it, runs it through the model, and sends back
#       a prediction as JSON
#
# WHAT IS REST?
#   REST (Representational State Transfer) is a style of API design.
#   The key idea: use standard HTTP methods (GET, POST, PUT, DELETE) to
#   perform actions on resources. Our API has two endpoints:
#     GET  /health  → "are you running?" check
#     POST /predict → "predict churn for this customer"
#
# HOW TO RUN LOCALLY (without Docker):
#   pip install -r requirements.txt
#   python src/train.py          (train the model first)
#   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# HOW TO TEST:
#   Open http://localhost:8000/docs in your browser — FastAPI auto-generates
#   an interactive UI (Swagger UI) where you can test all endpoints!
# =============================================================================

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.predict import initialize_predictor, get_predictor


# =============================================================================
# PYDANTIC SCHEMAS — Input/Output Validation
# =============================================================================
#
# WHAT IS PYDANTIC?
#   Pydantic is a data validation library. When a request comes in, FastAPI
#   uses our Pydantic schema to:
#     1. Check that all required fields are present
#     2. Check that each field has the correct type
#     3. Reject bad requests automatically with a 422 error and helpful message
#
# Without Pydantic, we'd have to write this validation manually. With it,
# we just define the shape of our data as a Python class.
# =============================================================================

class CustomerFeatures(BaseModel):
    """
    The input schema for a churn prediction request.

    Each field represents one customer attribute.
    Field(...) means the field is REQUIRED (no default value).
    Field(default=...) means it's OPTIONAL.

    The descriptions show up in the auto-generated /docs UI.
    """

    # How many months the customer has been with the company
    tenure: int = Field(..., ge=0, le=120, description="Number of months as a customer (0-120)")

    # Monthly bill amount
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount in USD")

    # Total amount paid over their entire tenure
    TotalCharges: float = Field(..., ge=0, description="Total charges paid over customer lifetime")

    # Contract type — strongly correlated with churn in practice
    Contract: str = Field(..., description="Contract type: 'Month-to-month', 'One year', 'Two year'")

    # Payment method
    PaymentMethod: str = Field(
        ...,
        description="Payment method: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'"
    )

    # Internet service type
    InternetService: str = Field(
        ...,
        description="Internet service type: 'DSL', 'Fiber optic', 'No'"
    )

    # Add-on services (all optional, default to "No")
    OnlineSecurity: str = Field(default="No", description="Online security add-on: 'Yes', 'No'")
    TechSupport: str = Field(default="No", description="Tech support add-on: 'Yes', 'No'")
    PaperlessBilling: str = Field(default="No", description="Paperless billing: 'Yes', 'No'")

    # Demographics
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Is senior citizen: 0 or 1")
    Dependents: str = Field(default="No", description="Has dependents: 'Yes', 'No'")
    Partner: str = Field(default="No", description="Has partner: 'Yes', 'No'")
    MultipleLines: str = Field(default="No", description="Multiple phone lines: 'Yes', 'No'")
    PhoneService: str = Field(default="Yes", description="Has phone service: 'Yes', 'No'")

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Reject invalid contract types before they reach the model."""
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"Contract must be one of: {valid}. Got: '{v}'")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        valid = {"DSL", "Fiber optic", "No"}
        if v not in valid:
            raise ValueError(f"InternetService must be one of: {valid}. Got: '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1572.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "PaperlessBilling": "Yes",
                "SeniorCitizen": 0,
                "Dependents": "No",
                "Partner": "No",
                "MultipleLines": "No",
                "PhoneService": "Yes",
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    The output schema for a prediction response.
    FastAPI validates that our response matches this shape before sending it.
    """
    churn: bool = Field(..., description="True if customer is predicted to churn")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk level: 'Low', 'Medium', or 'High'")
    message: str = Field(..., description="Human-readable interpretation of the prediction")


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    version: str


class BatchCustomerFeatures(BaseModel):
    """Input schema for batch predictions (multiple customers at once)."""
    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of customers to predict (max 100 per request)"
    )


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: list[PredictionResponse]
    total_customers: int
    high_risk_count: int


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: code before 'yield' runs at STARTUP,
    code after 'yield' runs at SHUTDOWN.

    We use this to load the ML model once when the server starts.
    Loading here means every request shares the same model object in memory.
    """
    # --- STARTUP ---
    print("[startup] Loading ML model into memory...")
    initialize_predictor()
    print("[startup] API is ready to serve predictions.")

    yield  # Server is running — handle requests

    # --- SHUTDOWN ---
    print("[shutdown] Server is shutting down.")


# Create the FastAPI application instance
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## Overview
    Predict whether a telecom customer will churn (cancel their service)
    based on their account and usage characteristics.

    ## How to use
    1. Send a **POST /predict** request with customer data as JSON
    2. Receive a prediction: churn probability and risk level

    ## Model
    Trained on the Telco Customer Churn dataset using XGBoost / Random Forest.
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check — verify the API is running",
)
def health_check():
    """
    Health check endpoint.

    Used by:
      - Docker to verify the container is healthy
      - Render/Railway to know when the app is ready to serve traffic
      - Monitoring systems to alert if the API goes down

    Returns 200 OK if everything is working.
    """
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict churn for a single customer",
)
def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.

    **Input:** Customer account features (see schema below)

    **Output:**
    - `churn`: True/False prediction
    - `churn_probability`: 0.0 (definitely stays) to 1.0 (definitely churns)
    - `risk_level`: Low / Medium / High
    - `message`: human-readable explanation

    **Example curl command:**
    ```bash
    curl -X POST "http://localhost:8000/predict" \\
         -H "Content-Type: application/json" \\
         -d '{
           "tenure": 24,
           "MonthlyCharges": 65.5,
           "TotalCharges": 1572.0,
           "Contract": "Month-to-month",
           "PaymentMethod": "Electronic check",
           "InternetService": "Fiber optic"
         }'
    ```
    """
    predictor = get_predictor()

    if predictor is None:
        # This shouldn't happen if startup completed, but we guard against it
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The server may still be starting up."
        )

    try:
        # Convert Pydantic model → dict → prediction
        result = predictor.predict(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        # Catch any unexpected errors and return a clean error message
        # (never expose raw stack traces to API clients in production)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Predict churn for multiple customers at once",
)
def predict_churn_batch(batch: BatchCustomerFeatures):
    """
    Predict churn for up to 100 customers in a single request.

    More efficient than calling /predict 100 times separately because
    it avoids the overhead of 100 HTTP round-trips.

    Returns predictions in the same order as the input list.
    """
    predictor = get_predictor()

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    predictions = []
    for customer in batch.customers:
        try:
            result = predictor.predict(customer.model_dump())
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed for a customer: {str(e)}"
            )

    high_risk_count = sum(1 for p in predictions if p.risk_level == "High")

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk_count,
    )


@app.get(
    "/",
    tags=["System"],
    summary="Root endpoint — API information",
)
def root():
    """
    Root endpoint. Returns basic API information.
    Useful for quickly confirming the API is live after deployment.
    """
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }

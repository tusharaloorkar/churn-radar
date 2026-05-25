# =============================================================================
# tests/test_api.py — API Tests
# =============================================================================
#
# WHAT IS TESTING AND WHY DOES IT MATTER?
#   Testing means writing code that automatically verifies your other code
#   works correctly. Instead of manually opening the browser and clicking
#   around every time you make a change, you run:
#       pytest tests/
#   ...and pytest tells you in seconds if anything broke.
#
# WHY WRITE TESTS FOR A PORTFOLIO PROJECT?
#   Most beginners skip tests. Including them signals to hiring managers:
#     - You write production-quality code, not just notebooks
#     - You think about edge cases and failure modes
#     - Your code is safe to change (tests catch regressions)
#
# HOW DO FASTAPI TESTS WORK?
#   FastAPI provides a TestClient that simulates HTTP requests WITHOUT
#   needing a real running server. It's fast, reliable, and runs in memory.
#
# HOW TO RUN:
#   From churn-predictor/ folder:
#     pytest tests/ -v
#   (-v = verbose, shows each test name and pass/fail)
# =============================================================================

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# --- Test Data ---
# A realistic customer that should predict HIGH churn risk
# (month-to-month contract, fiber optic, electronic check — known churn factors)
HIGH_CHURN_CUSTOMER = {
    "tenure": 2,
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
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

# A realistic customer that should predict LOW churn risk
# (two-year contract, bank transfer — committed, stable customers)
LOW_CHURN_CUSTOMER = {
    "tenure": 60,
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0,
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "PaperlessBilling": "No",
    "SeniorCitizen": 0,
    "Dependents": "Yes",
    "Partner": "Yes",
    "MultipleLines": "No",
    "PhoneService": "Yes",
}


# =============================================================================
# FIXTURE — Mock Predictor
# =============================================================================
#
# WHAT IS A FIXTURE?
#   A pytest fixture is a function that sets up state needed by tests.
#   @pytest.fixture means pytest will run this function before each test
#   that requests it (by including it as a parameter).
#
# WHY MOCK THE PREDICTOR?
#   Our tests should NOT require the model file to exist on disk.
#   We're testing the API layer (routing, validation, response format),
#   not the model itself. By mocking the predictor, we:
#     - Make tests fast (no disk I/O)
#     - Make tests isolated (pass even without a trained model)
#     - Test only what we intend to test (the API, not the ML logic)
#
# WHAT IS A MOCK?
#   A mock is a fake object that simulates a real object's behavior.
#   MagicMock() creates an object where any method call returns a
#   configurable value. We set predict() to return a specific result.
# =============================================================================

@pytest.fixture
def mock_predictor():
    """Create a mock predictor that returns controlled, predictable results."""
    predictor = MagicMock()

    # When predict() is called with a high-churn customer dict,
    # return a high-churn result. We use a default response for all calls.
    predictor.predict.return_value = {
        "churn": True,
        "churn_probability": 0.82,
        "risk_level": "High",
        "message": "This customer has a high likelihood of churning.",
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    """
    Create a FastAPI TestClient with the mock predictor injected.

    We use patch() to replace the real _predictor_instance (which requires
    a trained model file) with our mock. This lets us test the API in
    complete isolation from the ML model.
    """
    with patch("src.predict._predictor_instance", mock_predictor):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


# =============================================================================
# TESTS
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self, client):
        """
        The /health endpoint should always return HTTP 200 OK.
        If it returns anything else, our deployment platform thinks we're broken.
        """
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """
        The response body should have the expected fields.
        This ensures our HealthResponse schema matches what we return.
        """
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_healthy(self, client):
        """Status should be 'healthy' (not 'degraded', 'starting', etc.)"""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for POST /predict"""

    def test_valid_customer_returns_200(self, client):
        """A well-formed request should always return 200."""
        response = client.post("/predict", json=HIGH_CHURN_CUSTOMER)
        assert response.status_code == 200

    def test_prediction_response_has_required_fields(self, client):
        """
        The response must have all 4 fields.
        Missing fields would break any client consuming our API.
        """
        response = client.post("/predict", json=HIGH_CHURN_CUSTOMER)
        data = response.json()

        assert "churn" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "message" in data

    def test_churn_is_boolean(self, client):
        """churn must be True or False (not 0/1 or a string)."""
        response = client.post("/predict", json=HIGH_CHURN_CUSTOMER)
        assert isinstance(response.json()["churn"], bool)

    def test_churn_probability_is_between_0_and_1(self, client):
        """Probability must be a valid probability value."""
        response = client.post("/predict", json=HIGH_CHURN_CUSTOMER)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_is_valid(self, client):
        """risk_level must be one of the three defined levels."""
        response = client.post("/predict", json=HIGH_CHURN_CUSTOMER)
        risk = response.json()["risk_level"]
        assert risk in {"Low", "Medium", "High"}

    def test_missing_required_field_returns_422(self, client):
        """
        Sending a request without a required field should return 422 Unprocessable Entity.
        FastAPI + Pydantic handle this automatically — we just verify it works.
        422 is the correct HTTP status for invalid request data (not 400 or 500).
        """
        # Remove the required 'tenure' field
        bad_customer = {k: v for k, v in HIGH_CHURN_CUSTOMER.items() if k != "tenure"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_invalid_contract_type_returns_422(self, client):
        """An invalid contract value should be rejected before reaching the model."""
        bad_customer = {**HIGH_CHURN_CUSTOMER, "Contract": "Invalid Contract Type"}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, client):
        """Tenure can't be negative — Pydantic's ge=0 constraint should catch this."""
        bad_customer = {**HIGH_CHURN_CUSTOMER, "tenure": -5}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_negative_monthly_charges_returns_422(self, client):
        """Monthly charges can't be negative."""
        bad_customer = {**HIGH_CHURN_CUSTOMER, "MonthlyCharges": -10.0}
        response = client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_low_churn_customer_accepted(self, client, mock_predictor):
        """Low-churn customer input should also be accepted."""
        mock_predictor.predict.return_value = {
            "churn": False,
            "churn_probability": 0.12,
            "risk_level": "Low",
            "message": "This customer is likely to stay.",
        }
        response = client.post("/predict", json=LOW_CHURN_CUSTOMER)
        assert response.status_code == 200
        assert response.json()["churn"] is False


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch"""

    def test_batch_predict_returns_200(self, client):
        """Batch endpoint should accept a list of customers."""
        batch_payload = {"customers": [HIGH_CHURN_CUSTOMER, LOW_CHURN_CUSTOMER]}
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200

    def test_batch_response_has_correct_count(self, client):
        """total_customers should match the number of inputs."""
        batch_payload = {"customers": [HIGH_CHURN_CUSTOMER, HIGH_CHURN_CUSTOMER]}
        response = client.post("/predict/batch", json=batch_payload)
        data = response.json()
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2

    def test_empty_batch_returns_422(self, client):
        """An empty customer list should be rejected (min_length=1)."""
        response = client.post("/predict/batch", json={"customers": []})
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self, client):
        """Root should point users to the /docs UI."""
        data = response = client.get("/").json()
        assert "docs" in data

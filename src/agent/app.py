from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from .churn_tool import (
    predict_churn,
    simulate_churn_change,
    batch_predict,
    filter_high_risk_customers,
)

app = FastAPI(
    title="GAIA AI",
    description="FastAPI service for churn prediction, explanation, simulation, and batch scoring",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    customer_data: Dict[str, Any] = Field(..., description="Single customer record")


class SimulateRequest(BaseModel):
    customer_data: Dict[str, Any] = Field(..., description="Original customer data")
    changes: Dict[str, Any] = Field(..., description="Fields to change for simulation")


class BatchPredictRequest(BaseModel):
    customers: List[Dict[str, Any]] = Field(..., description="Customer list for batch prediction")


class HighRiskRequest(BaseModel):
    customers: List[Dict[str, Any]] = Field(..., description="Customer list")
    threshold: float = Field(0.7, description="High-risk probability threshold")
    limit: Optional[int] = Field(None, description="Maximum number of customers to return")


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "endpoints": ["/predict", "/simulate", "/batch-predict", "/high-risk", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    result = predict_churn(request.customer_data)

    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))

    return result


@app.post("/simulate")
def simulate(request: SimulateRequest):
    result = simulate_churn_change(request.customer_data, request.changes)

    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Simulation failed"))

    return result


@app.post("/batch-predict")
def batch_predict_api(request: BatchPredictRequest):
    try:
        print(f"[DEBUG] /batch-predict rows={len(request.customers)}")
        result = batch_predict(request.customers)
        print("[DEBUG] result:", result)

        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Batch prediction failed"))

        return result
    except Exception as e:
        print("[ERROR] /batch-predict exception:", repr(e))
        raise

@app.post("/high-risk")
def high_risk_api(request: HighRiskRequest):
    result = filter_high_risk_customers(
        customers=request.customers,
        threshold=request.threshold,
        limit=request.limit,
    )

    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "High-risk filtering failed"))

    return result
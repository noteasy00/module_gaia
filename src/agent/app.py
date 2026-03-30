from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from churn_tool import (
    predict_churn,
    simulate_churn_change,
    batch_predict,
    filter_high_risk_customers,
)

app = FastAPI(
    title="AI Customer Churn Prediction API",
    description="FastAPI service for churn prediction, simulation, batch prediction, and high-risk filtering",
    version="1.1.0"
)


class PredictRequest(BaseModel):
    customer_data: Dict[str, Any]


class SimulateRequest(BaseModel):
    customer_data: Dict[str, Any]
    changes: Dict[str, Any]


class BatchPredictRequest(BaseModel):
    customers: List[Dict[str, Any]]


class HighRiskRequest(BaseModel):
    customers: List[Dict[str, Any]]
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    limit: Optional[int] = Field(None, ge=1)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "endpoints": [
            "/predict",
            "/simulate",
            "/batch-predict",
            "/high-risk",
            "/health",
            "/docs",
        ],
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
def batch_predict_endpoint(request: BatchPredictRequest):
    result = batch_predict(request.customers)
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Batch prediction failed"))
    return result


@app.post("/high-risk")
def high_risk_endpoint(request: HighRiskRequest):
    result = filter_high_risk_customers(
        customers=request.customers,
        threshold=request.threshold,
        limit=request.limit,
    )
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "High-risk filtering failed"))
    return result

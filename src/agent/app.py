from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict

from churn_tool import predict_churn, simulate_churn_change

app = FastAPI(
    title="GAIA AI",
    description="FastAPI service for churn prediction, explanation, and simulation",
    version="스타크래프트 2.16.6 립버젼"
)


class PredictRequest(BaseModel):
    customer_data: Dict[str, Any] = Field(..., description="Single customer record")


class SimulateRequest(BaseModel):
    customer_data: Dict[str, Any] = Field(..., description="Original customer data")
    changes: Dict[str, Any] = Field(..., description="Fields to change for simulation")


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "endpoints": ["/predict", "/simulate", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


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
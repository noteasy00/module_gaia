import joblib
import pandas as pd

MODEL_PATH = "churn_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
THRESHOLD_PATH = "best_threshold.pkl"

DROP_COLS = [
    "CustomerID", "Count",
    "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason",
    "CLTV",
    "Churn Value"  
]


def preprocess_customer_data(customer_data: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])

    # Drop excluded columns if they appear
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors="ignore")

    # Convert Total Charges if provided
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Fill missing numeric values conservatively
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # One-hot encode 
    df = pd.get_dummies(df, drop_first=True)


    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder and remove unexpected extra columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def build_explanations(customer_data: dict, churn_probability: float) -> list:
    explanations = []

    contract = customer_data.get("Contract", "")
    internet_service = customer_data.get("Internet Service", "")
    payment_method = customer_data.get("Payment Method", "")
    tech_support = customer_data.get("Tech Support", "")
    online_security = customer_data.get("Online Security", "")
    tenure = customer_data.get("Tenure Months", 0)
    monthly_charges = customer_data.get("Monthly Charges", 0)

    if contract == "Month-to-month":
        explanations.append("Month-to-month contract tends to increase churn risk.")
    elif contract in ["One year", "Two year"]:
        explanations.append("Long-term contract tends to reduce churn risk.")

    if internet_service == "Fiber optic":
        explanations.append("Fiber optic service was associated with higher churn tendency.")
    if payment_method == "Electronic check":
        explanations.append("Electronic check was associated with higher churn risk.")
    if tech_support == "No":
        explanations.append("Lack of tech support may increase churn risk.")
    if online_security == "No":
        explanations.append("Lack of online security may increase churn risk.")

    if isinstance(tenure, (int, float)):
        if tenure < 12:
            explanations.append("Short tenure is often associated with higher churn risk.")
        elif tenure >= 24:
            explanations.append("Longer tenure tends to reduce churn risk.")

    if isinstance(monthly_charges, (int, float)) and monthly_charges >= 80:
        explanations.append("High monthly charges may contribute to churn risk.")

    if churn_probability >= 0.70:
        explanations.append("Overall profile indicates high churn risk.")
    elif churn_probability >= 0.40:
        explanations.append("Overall profile indicates moderate churn risk.")
    else:
        explanations.append("Overall profile indicates relatively low churn risk.")

    return explanations

def simulate_churn_change(customer_data: dict, changes: dict) -> dict:
    try:
        updated_customer = customer_data.copy()
        updated_customer.update(changes)

        before_result = predict_churn(customer_data)
        after_result = predict_churn(updated_customer)

        if not before_result.get("success", False):
            return before_result

        if not after_result.get("success", False):
            return after_result

        before_prob = before_result["churn_probability"]
        after_prob = after_result["churn_probability"]
        delta = round(after_prob - before_prob, 4)

        if delta < 0:
            impact = "Churn risk decreased"
        elif delta > 0:
            impact = "Churn risk increased"
        else:
            impact = "No significant change"

        return {
            "success": True,
            "original_customer": customer_data,
            "applied_changes": changes,
            "updated_customer": updated_customer,
            "before": before_result,
            "after": after_result,
            "probability_change": delta,
            "impact": impact
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
def predict_churn(customer_data: dict) -> dict:
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        threshold = joblib.load(THRESHOLD_PATH)

        X_input = preprocess_customer_data(customer_data, feature_columns)
        churn_probability = float(model.predict_proba(X_input)[0][1])
        prediction = int(churn_probability >= threshold)

        if churn_probability >= 0.70:
            risk_level = "High"
        elif churn_probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "success": True,
            "churn_probability": round(churn_probability, 4),
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "risk_level": risk_level,
            "threshold_used": float(threshold),
            "explanations": build_explanations(customer_data, churn_probability)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def batch_predict(customers: list) -> dict:
    try:
        results = []
        for cust in customers:
            res = predict_churn(cust)
            if res.get("success"):
                res["customer_data"] = cust
                results.append(res)
            else:
                results.append({"success": False, "error": res.get("error"), "customer_data": cust})
        
        return {
            "success": True,
            "total_processed": len(customers),
            "predictions": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def filter_high_risk_customers(customers: list, threshold: float, limit: int = None) -> dict:
    try:
        high_risk_list = []
        for cust in customers:
            res = predict_churn(cust)
            if res.get("success") and res.get("churn_probability", 0) >= threshold:
                res["customer_data"] = cust
                high_risk_list.append(res)
        
      
        high_risk_list = sorted(high_risk_list, key=lambda x: x["churn_probability"], reverse=True)
        
       
        if limit and isinstance(limit, int) and limit > 0:
            high_risk_list = high_risk_list[:limit]
            
        return {
            "success": True,
            "total_high_risk": len(high_risk_list),
            "threshold_applied": threshold,
            "filtered_customers": high_risk_list
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    sample_customer = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 89.5,
        "Total Charges": 450.2
    }

    result = predict_churn(sample_customer)
    print(result)

    
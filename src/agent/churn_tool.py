import joblib
import pandas as pd

MODEL_PATH = "churn_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
THRESHOLD_PATH = "best_threshold.pkl"

DROP_COLS = ["Churn"]  # target only


def preprocess_customer_data(customer_data: dict, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])

    # Remove target if accidentally included
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors="ignore")

    # One-hot encode exactly like training
    df = pd.get_dummies(df, drop_first=True)

    # Convert bool columns to int if any
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Add missing training columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reindex to exact training schema
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def build_explanations(customer_data: dict, churn_probability: float) -> list:
    explanations = []

    tenure = customer_data.get("tenure", 0)
    tenure_group = customer_data.get("Tenure_Group", "")
    monthly_charges = customer_data.get("MonthlyCharges", 0)
    senior_citizen = customer_data.get("SeniorCitizen", 0)
    internet_service = customer_data.get("InternetService", "")
    payment_method = customer_data.get("PaymentMethod", "")
    paperless_billing = customer_data.get("PaperlessBilling", "")
    tech_support = customer_data.get("TechSupport", "")
    service_engagement = customer_data.get("Service_Engagement", 0)
    contract = customer_data.get("Contract", "")

    if contract == "Month-to-month":
        explanations.append("Month-to-month contract tends to increase churn risk.")
    elif contract in ["One year", "Two year"]:
        explanations.append("Long-term contract tends to reduce churn risk.")

    if internet_service == "Fiber optic":
        explanations.append("Fiber optic service may be associated with higher churn risk.")

    if payment_method == "Electronic check":
        explanations.append("Electronic check customers may show higher churn tendency.")

    if paperless_billing == "Yes":
        explanations.append("Paperless billing was often observed among higher-risk customers.")

    if tech_support == "No":
        explanations.append("Lack of tech support may increase churn risk.")
    elif tech_support == "Yes":
        explanations.append("Having tech support may help reduce churn risk.")

    if isinstance(service_engagement, (int, float)):
        if service_engagement <= 1:
            explanations.append(f"Low service engagement ({service_engagement}) may increase churn risk.")
        elif service_engagement >= 4:
            explanations.append(f"Higher service engagement ({service_engagement}) may help reduce churn risk.")

    if isinstance(tenure, (int, float)):
        if tenure < 12:
            explanations.append(f"Short tenure ({tenure} months) is often associated with higher churn risk.")
        elif tenure >= 36:
            explanations.append(f"Longer tenure ({tenure} months) tends to reduce churn risk.")

    if tenure_group == "New(0-1yr)":
        explanations.append("Newer customers are often more likely to churn.")
    elif tenure_group == "Loyal(3yr+)":
        explanations.append("Long-term loyal customers are generally less likely to churn.")

    if isinstance(monthly_charges, (int, float)) and monthly_charges >= 80:
        explanations.append(f"Monthly charges ({monthly_charges}) are relatively high, which may increase churn risk.")

    if senior_citizen == 1:
        explanations.append("Senior citizen status may influence churn depending on service and pricing conditions.")

    if churn_probability >= 0.70:
        explanations.append("Overall profile indicates high churn risk.")
    elif churn_probability >= 0.40:
        explanations.append("Overall profile indicates moderate churn risk.")
    else:
        explanations.append("Overall profile indicates relatively low churn risk.")

    return explanations


def get_risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.70:
        return "High"
    elif churn_probability >= 0.40:
        return "Medium"
    return "Low"


def predict_churn(customer_data: dict) -> dict:
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        threshold = joblib.load(THRESHOLD_PATH)

        X_input = preprocess_customer_data(customer_data, feature_columns)
        churn_probability = float(model.predict_proba(X_input)[0][1])
        prediction = int(churn_probability >= threshold)

        return {
            "success": True,
            "churn_probability": round(churn_probability, 4),
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "risk_level": get_risk_level(churn_probability),
            "threshold_used": float(threshold),
            "explanations": build_explanations(customer_data, churn_probability)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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
def batch_predict(customers: list[dict]) -> dict:
    try:
        model, feature_columns, threshold = _load_artifacts()

        if not customers:
            return {"success": False, "error": "customers list is empty"}

        rows = []
        for idx, customer in enumerate(customers):
            row = customer.copy()
            row["_row_id"] = customer.get("customer_id", f"row_{idx}")
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        ids = raw_df["_row_id"].tolist()
        feature_df = raw_df.drop(columns=["_row_id"], errors="ignore")

        encoded = pd.get_dummies(feature_df, drop_first=True)
        bool_cols = encoded.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            encoded[bool_cols] = encoded[bool_cols].astype(int)

        for col in feature_columns:
            if col not in encoded.columns:
                encoded[col] = 0

        encoded = encoded.reindex(columns=feature_columns, fill_value=0)

        probs = model.predict_proba(encoded)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            prob = float(prob)
            pred = int(prob >= threshold)
            customer_data = customers[i]
            results.append({
                "customer_id": ids[i],
                "churn_probability": round(prob, 4),
                "prediction": pred,
                "prediction_label": "Churn" if pred == 1 else "No Churn",
                "risk_level": get_risk_level(prob),
                "explanations": build_explanations(customer_data, prob),
                "customer_data": customer_data
            })

        return {
            "success": True,
            "count": len(results),
            "threshold_used": threshold,
            "results": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def filter_high_risk_customers(customers: list[dict], threshold: float = 0.7, limit: int | None = None) -> dict:
    try:
        batch_result = batch_predict(customers)
        if not batch_result.get("success", False):
            return batch_result

        filtered = [
            r for r in batch_result["results"]
            if r["churn_probability"] >= threshold
        ]

        filtered.sort(key=lambda x: x["churn_probability"], reverse=True)

        if limit is not None:
            filtered = filtered[:limit]

        return {
            "success": True,
            "threshold": threshold,
            "count": len(filtered),
            "results": filtered
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


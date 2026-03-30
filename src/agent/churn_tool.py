import pickle
from functools import lru_cache
from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]   # module_gaia/
BUNDLE_PATH = BASE_DIR / "ckpt" / "xgb_top5_cv.pkl"
DROP_COLS = ["Churn"]


@lru_cache(maxsize=1)
def _load_bundle():
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def _prepare_raw_df(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df.copy()

    # target 제거
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors="ignore")

    feature_columns = bundle["feature_columns"]
    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])

    # 학습 시 기대한 raw feature column이 없으면 채워주기
    for col in feature_columns:
        if col not in df.columns:
            if col in numeric_features:
                df[col] = 0
            else:
                df[col] = "Unknown"

    # extra column 제거 + 순서 정렬
    df = df.reindex(columns=feature_columns)

    # bool 처리
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # 숫자형 결측 보정
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 범주형 결측 보정
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def _predict_proba_df(df: pd.DataFrame):
    bundle = _load_bundle()
    raw_df = _prepare_raw_df(df, bundle)

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    X_processed = preprocessor.transform(raw_df)
    probs = model.predict_proba(X_processed)[:, 1]

    return probs, threshold, bundle


def build_explanations(customer_data: dict, churn_probability: float) -> list:
    explanations = []

    # 데이터 추출
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

    # 1. 계약 형태 관련
    if contract == "Month-to-month":
        explanations.append("단기 계약(Month-to-month)은 이탈 위험을 높이는 주요 요인입니다.")
    elif contract in ["One year", "Two year"]:
        explanations.append("장기 계약 상태로, 이탈 위험을 낮추는 데 기여하고 있습니다.")

    # 2. 서비스 특성 관련
    if internet_service == "Fiber optic":
        explanations.append("광랜(Fiber optic) 서비스 이용자는 품질에 민감하여 이탈 가능성이 높게 관찰됩니다.")

    if payment_method == "Electronic check":
        explanations.append("전자 수표(Electronic check) 결제 방식 사용 고객군에서 높은 이탈 경향이 나타납니다.")

    if paperless_billing == "Yes":
        explanations.append("종이 없는 고지서(Paperless billing) 이용 고객은 고위험군에서 자주 발견되는 특징입니다.")

    # 3. 기술 지원 관련
    if tech_support == "No":
        explanations.append("전문 기술 지원(TechSupport) 부재는 서비스 불만족 및 이탈로 이어질 수 있습니다.")
    elif tech_support == "Yes":
        explanations.append("전문 기술 지원 이용은 고객 유지에 긍정적인 영향을 미칩니다.")

    # 4. 서비스 인게이지먼트 관련
    if isinstance(service_engagement, (int, float)):
        if service_engagement <= 1:
            explanations.append(f"낮은 서비스 결합도({service_engagement}개)는 브랜드 충성도를 약화시키는 요인입니다.")
        elif service_engagement >= 4:
            explanations.append(f"높은 서비스 결합도({service_engagement}개)는 강력한 이탈 방지 요인으로 작용합니다.")

    # 5. 가입 기간(Tenure) 관련
    if isinstance(tenure, (int, float)):
        if tenure < 12:
            explanations.append(f"가입 초기 단계({tenure}개월)로, 서비스 정착을 위한 집중 관리가 필요합니다.")
        elif tenure >= 36:
            explanations.append(f"장기 이용 고객({tenure}개월)으로, 높은 충성도를 유지하고 있습니다.")

    # 6. 월 요금 관련
    if isinstance(monthly_charges, (int, float)) and monthly_charges >= 80:
        explanations.append(f"상대적으로 높은 월 요금(${monthly_charges})이 이탈 위험을 증가시킬 수 있습니다.")

    # 7. 고령자 여부
    if senior_citizen == 1:
        explanations.append("고령 고객층의 특성에 맞는 맞춤형 혜택 안내가 권장됩니다.")

    # 8. 종합 결과 문구
    if churn_probability >= 0.70:
        explanations.append("전체적인 프로필이 **매우 높은 이탈 위험**을 나타내고 있습니다.")
    elif churn_probability >= 0.40:
        explanations.append("전체적인 프로필이 **보통 수준의 이탈 위험**을 나타내고 있습니다.")
    else:
        explanations.append("전체적인 프로필이 **비교적 낮은 이탈 위험**을 나타내고 있습니다.")

    return explanations


def get_risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.70:
        return "High"
    elif churn_probability >= 0.40:
        return "Medium"
    return "Low"


def predict_churn(customer_data: dict) -> dict:
    try:
        df = pd.DataFrame([customer_data])
        probs, threshold, bundle = _predict_proba_df(df)

        churn_probability = float(probs[0])
        prediction = int(churn_probability >= threshold)

        return {
            "success": True,
            "model_name": bundle.get("model_name", "xgb_top5_cv"),
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

        # 텍스트 메시지 한글화
        if delta < 0:
            impact = "이탈 위험도가 감소했습니다."
        elif delta > 0:
            impact = "이탈 위험도가 증가했습니다."
        else:
            impact = "유의미한 변화가 없습니다."

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
        if not customers:
            return {"success": False, "error": "customers list is empty"}

        rows = []
        ids = []

        for idx, customer in enumerate(customers):
            row = customer.copy()
            row_id = row.get("customer_id", f"row_{idx}")
            ids.append(row_id)
            rows.append(row)

        df = pd.DataFrame(rows)
        probs, threshold, bundle = _predict_proba_df(df)

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
            "model_name": bundle.get("model_name", "xgb_top5_cv"),
            "count": len(results),
            "threshold_used": float(threshold),
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

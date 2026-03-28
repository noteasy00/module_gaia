import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =========================================================
# 0. 설정 세팅
# =========================================================
FULL_PATH = "data/telco_churn_full.csv"
TOP5_PATH = "data/telco_churn_top5_with_engineering_2.csv"

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

RANDOM_STATE = 42

target_col = "Churn"

# =========================================================
# 1. 공통 유틸
# =========================================================

def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """
    숫자형 / 범주형 자동 분리 후 전처리기 생성
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", ohe, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def tune_threshold(y_true: np.ndarray, prob: np.ndarray,
                   thresholds: np.ndarray | None = None) -> tuple[float, float]:
    """
    validation F1 기준 최적 threshold 탐색
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.02)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1


def evaluate_classifier(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """
    확률 + threshold 기반 평가
    """
    pred = (prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, zero_division=0, output_dict=True)
    }
    return metrics


def train_one_dataset(csv_path: str, model_name: str) -> dict:
    """
    하나의 CSV에 대해:
    - 데이터 로드
    - train/valid/test 분리
    - XGBClassifier 학습
    - valid threshold tuning
    - test 평가
    """
    print("\n" + "=" * 80)
    print(f"[학습 시작] {model_name} | file={csv_path}")

    df = pd.read_csv(csv_path)
    print("shape:", df.shape)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # valid / test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("train:", X_train.shape, "valid:", X_valid.shape, "test:", X_test.shape)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_valid_enc = preprocessor.transform(X_valid)
    X_test_enc = preprocessor.transform(X_test)

    print("encoded train shape:", X_train_enc.shape)
    print("numeric:", numeric_features)
    print("categorical:", categorical_features)

    # 불균형 데이터 보정
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train_enc, y_train)

    valid_prob = model.predict_proba(X_valid_enc)[:, 1]
    best_thr, best_valid_f1 = tune_threshold(y_valid.to_numpy(), valid_prob)

    print(f"best threshold on valid: {best_thr:.2f}")
    print(f"best valid F1: {best_valid_f1:.4f}")

    test_prob = model.predict_proba(X_test_enc)[:, 1]
    test_metrics = evaluate_classifier(y_test.to_numpy(), test_prob, best_thr)

    print("\n[test metrics]")
    print({
        "f1": round(test_metrics["f1"], 4),
        "precision": round(test_metrics["precision"], 4),
        "recall": round(test_metrics["recall"], 4),
        "roc_auc": round(test_metrics["roc_auc"], 4),
        "pr_auc": round(test_metrics["pr_auc"], 4),
        "threshold": round(test_metrics["threshold"], 2)
    })

    feature_columns = X.columns.tolist()

    result = {
        "model_name": model_name,
        "csv_path": csv_path,
        "target_col": target_col,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "threshold": best_thr,
        "valid_f1": best_valid_f1,
        "test_metrics": test_metrics,
        "preprocessor": preprocessor,
        "model": model
    }

    return result


def save_bundle(result: dict, save_path: str):
    """
    추론용 번들 저장
    """
    bundle = {
        "model_name": result["model_name"],
        "target_col": result["target_col"],
        "feature_columns": result["feature_columns"],
        "numeric_features": result["numeric_features"],
        "categorical_features": result["categorical_features"],
        "threshold": result["threshold"],
        "test_metrics": result["test_metrics"],
        "preprocessor": result["preprocessor"],
        "model": result["model"]
    }
    joblib.dump(bundle, save_path)


# =========================================================
# 2. 전체 피처 모델 학습
# =========================================================
full_result = train_one_dataset(FULL_PATH, model_name="xgb_full")

# =========================================================
# 3. top5 피처 모델 학습
# =========================================================
top5_result = train_one_dataset(TOP5_PATH, model_name="xgb_top5")

# =========================================================
# 4. 결과 비교
# =========================================================
summary_rows = []

for res in [full_result, top5_result]:
    m = res["test_metrics"]
    summary_rows.append({
        "model_name": res["model_name"],
        "num_input_features": len(res["feature_columns"]),
        "valid_f1": res["valid_f1"],
        "test_f1": m["f1"],
        "test_precision": m["precision"],
        "test_recall": m["recall"],
        "test_roc_auc": m["roc_auc"],
        "test_pr_auc": m["pr_auc"],
        "threshold": res["threshold"]
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["test_f1", "test_roc_auc", "test_pr_auc"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 80)
print("[최종 비교 결과]")
print(summary_df)

summary_df.to_csv(os.path.join(SAVE_DIR, "model_comparison_summary.csv"), index=False)

# =========================================================
# 5. 최종 모델 선택 및 저장
# =========================================================
best_name = summary_df.iloc[0]["model_name"]
best_result = full_result if best_name == "xgb_full" else top5_result

final_model_path = os.path.join(SAVE_DIR, "best_telco_churn_model.joblib")
save_bundle(best_result, final_model_path)

print("\n" + "=" * 80)
print("[최종 선택 모델]")
print("best model:", best_name)
print("saved to:", final_model_path)

# JSON 요약도 저장
final_summary = {
    "best_model": best_name,
    "threshold": float(best_result["threshold"]),
    "feature_columns": best_result["feature_columns"],
    "test_metrics": {
        "f1": float(best_result["test_metrics"]["f1"]),
        "precision": float(best_result["test_metrics"]["precision"]),
        "recall": float(best_result["test_metrics"]["recall"]),
        "roc_auc": float(best_result["test_metrics"]["roc_auc"]),
        "pr_auc": float(best_result["test_metrics"]["pr_auc"])
    }
}

with open(os.path.join(SAVE_DIR, "best_model_summary.json"), "w", encoding="utf-8") as f:
    json.dump(final_summary, f, ensure_ascii=False, indent=2)

print("summary saved to:", os.path.join(SAVE_DIR, "best_model_summary.json"))
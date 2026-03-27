import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

import shap

import os
save_dir = "ML-Regression/figures"
os.makedirs(save_dir, exist_ok=True)

# =========================================
# 1. 데이터 로드
# =========================================
file_path = "data/Telco-Customer-Churn.csv"   # 필요시 경로 수정
df = pd.read_csv(file_path)

print("원본 shape:", df.shape)
print(df.head())

# =========================================
# 2. 전처리
# =========================================
# TotalCharges 숫자형 변환
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 결측 제거
df = df.dropna(subset=["TotalCharges"]).copy()

# 타깃: churn을 0/1로 변환
df["ChurnNum"] = df["Churn"].map({"No": 0, "Yes": 1})

# feature / target 분리
# 예측 모델에서는 정보 손실 줄이려고 customerID, Churn만 제외하고 사용
X = df.drop(columns=["customerID", "Churn", "ChurnNum"])
y = df["ChurnNum"]

print("\n전처리 후 shape:", df.shape)
print("타깃 분포:")
print(y.value_counts(normalize=True).round(4))

# 숫자형 / 범주형 분리
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\n숫자형 컬럼:", numeric_features)
print("범주형 컬럼:", categorical_features)

# =========================================
# 3. 학습 / 테스트 분리
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# 4. 전처리기
# =========================================
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", ohe, categorical_features)
    ]
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# SHAP과 해석 편하게 DataFrame으로 변환
X_train_df = pd.DataFrame(X_train_enc, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_enc, columns=feature_names, index=X_test.index)

print("\n인코딩 후 shape:", X_train_df.shape)

# =========================================
# 5. 회귀 모델 후보 정의
# =========================================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.0005, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTreesRegressor": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        random_state=42
    )
}

# =========================================
# 6. 모델 학습 및 평가
# =========================================
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_df, y_train)
    pred = model.predict(X_test_df)

    # 회귀 예측값은 0~1 밖으로 나갈 수 있으니
    # 해석/auc용으로 clip 버전도 같이 사용
    pred_clip = np.clip(pred, 0, 1)

    mae = mean_absolute_error(y_test, pred_clip)
    rmse = np.sqrt(mean_squared_error(y_test, pred_clip))
    r2 = r2_score(y_test, pred_clip)
    auc = roc_auc_score(y_test, pred_clip)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "ROC_AUC": auc
    })

    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(
    by=["RMSE", "ROC_AUC"],
    ascending=[True, False]
).reset_index(drop=True)

print("\n[모델 비교 결과]")
print(results_df)

# =========================================
# 7. 최종 모델 선택
# =========================================
# 기본적으로 RMSE 가장 낮은 모델 선택
best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print(f"\n최종 선택 모델: {best_model_name}")

# 테스트 예측
best_pred = np.clip(best_model.predict(X_test_df), 0, 1)

# 예측값 분포 확인
plt.figure(figsize=(8, 5))
plt.hist(best_pred, bins=30)
plt.title(f"Prediction Distribution - {best_model_name}")
plt.xlabel("Predicted Churn Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{save_dir}/prediction_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================
# 8. SHAP 분석
# =========================================
# 너무 오래 걸리는 걸 막기 위해 일부 샘플 사용
background_size = min(300, len(X_train_df))
explain_size = min(500, len(X_test_df))

X_background = X_train_df.sample(background_size, random_state=42)
X_explain = X_test_df.sample(explain_size, random_state=42)

print("\nSHAP 분석 시작...")
print("background_size:", len(X_background))
print("explain_size:", len(X_explain))

# 모델 유형에 따라 explainer 선택
if best_model_name in ["RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_explain)

    # shap_values가 list로 나오는 경우 방지
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # summary plot
    shap.summary_plot(shap_values, X_explain, max_display=15, show=False)
    plt.savefig(f"{save_dir}/shap_summary_{best_model_name}.png", dpi=300, bbox_inches="tight")

    # bar plot
    shap.summary_plot(shap_values, X_explain, plot_type="bar", max_display=15, show=False)
    plt.savefig(f"{save_dir}/shap_bar_{best_model_name}.png", dpi=300, bbox_inches="tight")

    # 평균 절대 shap 값 정리
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

else:
    # 선형모델 계열은 generic explainer 사용
    explainer = shap.Explainer(best_model, X_background)
    shap_exp = explainer(X_explain)

    shap.plots.beeswarm(shap_exp, max_display=15)
    shap.plots.bar(shap_exp, max_display=15)

    shap_importance = pd.DataFrame({
        "feature": X_explain.columns,
        "mean_abs_shap": np.abs(shap_exp.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

print("\n[SHAP 중요도 Top 20]")
print(shap_importance.head(20))

# =========================================
# 9. 중요 피처 시각화
# =========================================
top_n = 15
top_features = shap_importance.head(top_n)

plt.figure(figsize=(12, 8))
plt.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
plt.title(f"Top {top_n} SHAP Features - {best_model_name}")
plt.xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.savefig(f"{save_dir}/shap_top_features_{best_model_name}.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================
# 10. 결과 저장
# =========================================
joblib.dump({
    "preprocessor": preprocessor,
    "model": best_model,
    "feature_names": list(feature_names),
    "best_model_name": best_model_name,
    "results_df": results_df
}, "ML-Regression/best_regression_model_with_preprocessor.pkl")

results_df.to_csv("ML-Regression/regression_model_comparison.csv", index=False)
shap_importance.to_csv("ML-Regression/best_model_shap_importance.csv", index=False)

print("\n저장 완료:")
print("- best_regression_model_with_preprocessor.pkl")
print("- regression_model_comparison.csv")
print("- best_model_shap_importance.csv")
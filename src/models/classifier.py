import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

# =========================================================
# 0. 설정 세팅
# =========================================================
FULL_PATH = "data/telco_churn_full.csv"
TOP5_PATH = "data/telco_churn_top5_with_engineering_2.csv"

SAVE_DIR = "ckpt"
OUTPUT_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

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


def tune_threshold(y_true, prob, thresholds=None):
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


def evaluate_classifier(y_true, prob, threshold):
    """
    확률 + threshold 기반 평가
    """
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def build_xgb_classifier(y_train):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        learning_rate= 0.05,
        max_depth=3,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    return model


#cm 시각화
def save_confusion_matrix_figure(y_true, prob, threshold, save_path, title="Confusion Matrix"):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()



# (메인) 5-fold CV로 안정 threshold 찾아 학습
def train_with_cv_threshold(csv_path: str, model_name: str) -> dict:
    print("\n" + "=" * 80)
    print(f"[학습 시작] {model_name} | file={csv_path}")

    df = pd.read_csv(csv_path)
    print("raw shape:", df.shape)

    y = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    # 먼저 test 홀드아웃
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("dev:", X_dev.shape, "test:", X_test.shape)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    fold_thresholds = []
    oof_prob = np.zeros(len(X_dev), dtype=float)

    X_dev = X_dev.reset_index(drop=True)
    y_dev = y_dev.reset_index(drop=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        X_tr = X_dev.iloc[tr_idx].copy()
        y_tr = y_dev.iloc[tr_idx].copy()
        X_va = X_dev.iloc[va_idx].copy()
        y_va = y_dev.iloc[va_idx].copy()

        preprocessor, numeric_features, categorical_features = build_preprocessor(X_tr)

        X_tr_enc = preprocessor.fit_transform(X_tr)
        X_va_enc = preprocessor.transform(X_va)

        model = build_xgb_classifier(y_tr)
        model.fit(X_tr_enc, y_tr)

        va_prob = model.predict_proba(X_va_enc)[:, 1]
        oof_prob[va_idx] = va_prob

        best_thr, best_f1 = tune_threshold(y_va.to_numpy(), va_prob)
        fold_thresholds.append(best_thr)

        fold_metrics = evaluate_classifier(y_va.to_numpy(), va_prob, best_thr)

        fold_rows.append({
            "fold": fold,
            "threshold": best_thr,
            "accuracy": fold_metrics["accuracy"],
            "f1": fold_metrics["f1"],
            "precision": fold_metrics["precision"],
            "recall": fold_metrics["recall"],
            "roc_auc": fold_metrics["roc_auc"],
            "pr_auc": fold_metrics["pr_auc"]
        })

        print(f"\n[fold {fold}]")
        print({
            "threshold": round(best_thr, 3),
            "accuracy": fold_metrics["accuracy"],
            "f1": round(fold_metrics["f1"], 4),
            "precision": round(fold_metrics["precision"], 4),
            "recall": round(fold_metrics["recall"], 4),
            "roc_auc": round(fold_metrics["roc_auc"], 4),
            "pr_auc": round(fold_metrics["pr_auc"], 4)
        })

    cv_results_df = pd.DataFrame(fold_rows)

    # threshold 집계
    threshold_mean = float(np.mean(fold_thresholds))
    threshold_median = float(np.median(fold_thresholds))

    # median 추천 (이상치에 덜 민감)
    stable_threshold = threshold_median

    print("\n" + "-" * 80)
    print("[CV threshold summary]")
    print("fold thresholds:", [round(x, 3) for x in fold_thresholds])
    print("mean threshold  :", round(threshold_mean, 4))
    print("median threshold:", round(threshold_median, 4))
    print("selected stable threshold:", round(stable_threshold, 4))

    # OOF 전체 점수
    oof_auc = roc_auc_score(y_dev, oof_prob)
    oof_pr_auc = average_precision_score(y_dev, oof_prob)
    oof_pred = (oof_prob >= stable_threshold).astype(int)
    oof_f1 = f1_score(y_dev, oof_pred, zero_division=0)

    print("\n[OOF metrics with stable threshold]")
    print({
        "oof_f1": round(oof_f1, 4),
        "oof_roc_auc": round(oof_auc, 4),
        "oof_pr_auc": round(oof_pr_auc, 4)
    })

    # dev 전체로 다시 학습
    final_preprocessor, numeric_features, categorical_features = build_preprocessor(X_dev)
    X_dev_enc = final_preprocessor.fit_transform(X_dev)
    X_test_enc = final_preprocessor.transform(X_test)

    final_model = build_xgb_classifier(y_dev)
    final_model.fit(X_dev_enc, y_dev)

    test_prob = final_model.predict_proba(X_test_enc)[:, 1]
    test_metrics = evaluate_classifier(y_test.to_numpy(), test_prob, stable_threshold)

    print("\n[test metrics with stable threshold]")
    print({
        "accuracy": round(test_metrics["accuracy"], 4),
        "f1": round(test_metrics["f1"], 4),
        "precision": round(test_metrics["precision"], 4),
        "recall": round(test_metrics["recall"], 4),
        "roc_auc": round(test_metrics["roc_auc"], 4),
        "pr_auc": round(test_metrics["pr_auc"], 4),
        "threshold": round(test_metrics["threshold"], 4)
    })

    save_confusion_matrix_figure(
    y_test.to_numpy(),
    test_prob,
    stable_threshold,
    os.path.join(OUTPUT_DIR, f"{model_name}_test_confusion_matrix.png"),
    title=f"{model_name} Test Confusion Matrix"
    )

    return {
        "model_name": model_name,
        "csv_path": csv_path,
        "target_col": target_col,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "threshold": stable_threshold,
        "cv_thresholds": fold_thresholds,
        "cv_threshold_mean": threshold_mean,
        "cv_threshold_median": threshold_median,
        "cv_results": cv_results_df,
        "test_metrics": test_metrics,
        "preprocessor": final_preprocessor,
        "model": final_model
    }



def save_pickle_bundle(result: dict, save_path: str):
    """
    UI팀 / OpenAI agent팀이 바로 쓸 수 있게
    필요한 정보 전부 묶어서 .pkl 저장
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

    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)



# =========================================================
# 2. 모델 학습
# =========================================================
# 전체 피처 
full_result = train_with_cv_threshold(FULL_PATH, "xgb_full_cv")

# top5 피처 
top5_result = train_with_cv_threshold(TOP5_PATH, "xgb_top5_cv")



# =========================================================
# 3. 결과 비교
# =========================================================
summary_df = pd.DataFrame([
    {
        "model_name": full_result["model_name"],
        "num_input_features": len(full_result["feature_columns"]),
        "cv_threshold_mean": full_result["cv_threshold_mean"],
        "cv_threshold_median": full_result["cv_threshold_median"],
        "selected_threshold": full_result["threshold"],
        "test_accuracy": full_result["test_metrics"]["accuracy"],
        "test_f1": full_result["test_metrics"]["f1"],
        "test_precision": full_result["test_metrics"]["precision"],
        "test_recall": full_result["test_metrics"]["recall"],
        "test_roc_auc": full_result["test_metrics"]["roc_auc"],
        "test_pr_auc": full_result["test_metrics"]["pr_auc"]
    },
    {
        "model_name": top5_result["model_name"],
        "num_input_features": len(top5_result["feature_columns"]),
        "cv_threshold_mean": top5_result["cv_threshold_mean"],
        "cv_threshold_median": top5_result["cv_threshold_median"],
        "selected_threshold": top5_result["threshold"],
        "test_accuracy": top5_result["test_metrics"]["accuracy"],
        "test_f1": top5_result["test_metrics"]["f1"],
        "test_precision": top5_result["test_metrics"]["precision"],
        "test_recall": top5_result["test_metrics"]["recall"],
        "test_roc_auc": top5_result["test_metrics"]["roc_auc"],
        "test_pr_auc": top5_result["test_metrics"]["pr_auc"]
    }
]).sort_values(
    # 이탈 고객을 잘 찾는 능력이 중요. PR-AUC와 F1 Score 집중
    by=["test_pr_auc", "test_f1", "test_recall", "test_roc_auc"],
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 80)
print("[최종 비교 결과]")
print(summary_df)

summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_threshold_model_comparison.csv"), index=False)
print("-", os.path.join(OUTPUT_DIR, "cv_threshold_model_comparison.csv"))


# =========================================================
# 3. 결과 저장
# =========================================================

# fold별 csv
full_result["cv_results"].to_csv(os.path.join(OUTPUT_DIR, "xgb_full_cv_fold_results.csv"), index=False)
top5_result["cv_results"].to_csv(os.path.join(OUTPUT_DIR, "xgb_top5_cv_fold_results.csv"), index=False)

# pkl 저장
save_pickle_bundle(full_result, os.path.join(SAVE_DIR, "xgb_full_cv.pkl"))
save_pickle_bundle(top5_result, os.path.join(SAVE_DIR, "xgb_top5_cv.pkl"))

best_name = summary_df.iloc[0]["model_name"]
#best_result = full_result if best_name == "xgb_full_cv" else top5_result
#save_pickle_bundle(best_result, os.path.join(SAVE_DIR, "final_selected_model_cv.pkl"))

print("\n 모델 저장 완료 ")
print("-", os.path.join(SAVE_DIR, "xgb_full_cv.pkl"))
print("-", os.path.join(SAVE_DIR, "xgb_top5_cv.pkl"))
#print("-", os.path.join(SAVE_DIR, "final_selected_model_cv.pkl"))

print("\n << ui, agent, tool팀 xgb_top5_cv.pkl 사용하면 됩니다 (학습데이터: telco_churn_top5_with_engineering_2.csv) >>")

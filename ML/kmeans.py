import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import TruncatedSVD
import joblib

# =========================================
# 1. 데이터 로드
# =========================================
file_path = "data/Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
print(df.head())

# TotalCharges 숫자형 변환 + 결측 제거
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).copy()

# 원본 보존
raw_df = df.copy()

# =========================================
# 2. 군집화용 피처 선택
# =========================================
# 핵심: 가입기간, 월요금, 계약형태
# 보조: 결제방식, 인터넷유형
cluster_features = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "PaymentMethod",
    "InternetService"
]

X = df[cluster_features].copy()

numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "PaymentMethod", "InternetService"]

print("\n[군집화 입력 피처]")
print(cluster_features)

print("\n숫자형 컬럼:", numeric_features)
print("범주형 컬럼:", categorical_features)

# =========================================
# 3. 전처리
# =========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

print("\n전처리 후 데이터 shape:", X_processed.shape)

# =========================================
# 4. k 후보 평가
#    - 엘보우(inertia)
#    - 실루엣
#    - churn gap
# =========================================
def get_churn_gap(df, labels):
    temp = df.copy()
    temp["Cluster"] = labels
    churn_rate = (temp["Churn"] == "Yes").groupby(temp["Cluster"]).mean()
    gap = churn_rate.max() - churn_rate.min()
    return gap, churn_rate

k_range = range(2, 7)

results = []
inertias = []
silhouette_scores = []
churn_gaps = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)

    inertia = kmeans.inertia_
    sil = silhouette_score(X_processed, labels)
    gap, churn_rate = get_churn_gap(raw_df, labels)

    inertias.append(inertia)
    silhouette_scores.append(sil)
    churn_gaps.append(gap)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": sil,
        "churn_gap": gap
    })

    print(f"\nk={k}")
    print(f"  inertia     : {inertia:.2f}")
    print(f"  silhouette  : {sil:.4f}")
    print(f"  churn_gap   : {gap:.4f}")
    print(f"  churn_by_cluster:\n{churn_rate.round(3)}")

results_df = pd.DataFrame(results)

print("\n[k 비교 결과]")
print(results_df)

# =========================================
# 5. 그래프
# =========================================
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by k")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), churn_gaps, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Churn Gap (max - min)")
plt.title("Churn Gap by k")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================
# 6. 최종 k 선택
# =========================================
# 여기서 elbow + churn gap + 해석 가능성을 보고 직접 선택
# 3 또는 4 확인
final_k = 3

print(f"\n최종 선택 k: {final_k}")

final_kmeans = KMeans(n_clusters=final_k, init="k-means++", random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_processed)

raw_df["Cluster"] = final_labels

# =========================================
# 7. 실루엣 샘플 시각화
# =========================================
sample_silhouette_values = silhouette_samples(X_processed, final_labels)
avg_silhouette = silhouette_score(X_processed, final_labels)

plt.figure(figsize=(10, 6))
y_lower = 10

for i in range(final_k):
    vals = sample_silhouette_values[final_labels == i]
    vals.sort()

    size_i = vals.shape[0]
    y_upper = y_lower + size_i

    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        vals,
        alpha=0.7
    )

    plt.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10

plt.axvline(x=avg_silhouette, linestyle="--", label=f"avg={avg_silhouette:.4f}")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster")
plt.title(f"Silhouette Plot (k={final_k})")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# 8. 2차원 시각화 + 축 해석
#    sparse 데이터이므로 PCA 대신 TruncatedSVD 사용
# =========================================
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_processed)

explained = svd.explained_variance_ratio_
print("\n[SVD explained variance ratio]")
print(f"Component 1: {explained[0]:.4f}")
print(f"Component 2: {explained[1]:.4f}")
print(f"합계       : {(explained[0] + explained[1]):.4f}")

loadings = pd.DataFrame(
    svd.components_.T,
    index=feature_names,
    columns=["Comp1", "Comp2"]
)

print("\n[Comp1에 크게 기여한 상위 변수]")
print(loadings["Comp1"].sort_values(key=np.abs, ascending=False).head(10))

print("\n[Comp2에 크게 기여한 상위 변수]")
print(loadings["Comp2"].sort_values(key=np.abs, ascending=False).head(10))

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=final_labels, alpha=0.7)
plt.xlabel(f"Component 1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"Component 2 ({explained[1]*100:.1f}%)")
plt.title(f"K-Means Clusters Visualized in 2D (k={final_k})")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# =========================================
# 9. 군집 프로파일링
# =========================================
print("\n[군집별 고객 수]")
print(raw_df["Cluster"].value_counts().sort_index())

print("\n[군집별 숫자형 평균]")
numeric_profile = raw_df.groupby("Cluster")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(2)
print(numeric_profile)

print("\n[군집별 Churn 비율]")
churn_profile = pd.crosstab(raw_df["Cluster"], raw_df["Churn"], normalize="index").round(3)
print(churn_profile)

target_cats = ["Contract", "PaymentMethod", "InternetService"]

for col in target_cats:
    print(f"\n[군집별 {col} 비율]")
    tab = pd.crosstab(raw_df["Cluster"], raw_df[col], normalize="index").round(3)
    print(tab)

print("\n[전체 평균 대비 군집 평균 차이]")
overall_mean = raw_df[["tenure", "MonthlyCharges", "TotalCharges"]].mean()
diff_profile = raw_df.groupby("Cluster")[["tenure", "MonthlyCharges", "TotalCharges"]].mean() - overall_mean
print(diff_profile.round(2))

# =========================================
# 10. 결과 해석
# =========================================
for c in sorted(raw_df["Cluster"].unique()):
    row = raw_df[raw_df["Cluster"] == c]

    top_contract = row["Contract"].value_counts(normalize=True).idxmax()
    top_internet = row["InternetService"].value_counts(normalize=True).idxmax()
    top_payment = row["PaymentMethod"].value_counts(normalize=True).idxmax()
    churn_rate = (row["Churn"] == "Yes").mean()
    avg_tenure = row["tenure"].mean()
    avg_monthly = row["MonthlyCharges"].mean()

    print(f"\n=== Cluster {c} 해석 ===")
    print(f"- 대표 계약 형태: {top_contract}")
    print(f"- 대표 인터넷 서비스: {top_internet}")
    print(f"- 대표 결제 방식: {top_payment}")
    print(f"- 평균 가입기간: {avg_tenure:.1f}")
    print(f"- 평균 월요금: {avg_monthly:.1f}")
    print(f"- 이탈률: {churn_rate:.3f}")

# =========================================
# 11. 결과 저장
# =========================================
raw_df.to_csv("ML/result/telco_clustered_result.csv", index=False)

joblib.dump({
    "preprocessor": preprocessor,
    "kmeans_model": final_kmeans,
    "cluster_features": cluster_features,
    "best_k": final_k
}, "ML/telco_kmeans_model.pkl")

print("\n저장 완료:")
print("- ML/result/telco_clustered_result.csv")
print("- ML/telco_kmeans_model.pkl")
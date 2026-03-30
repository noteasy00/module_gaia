import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load("model.pkl")
preprocessor = joblib.load("prepocessor.pkl")


def predict_from_csv(file_path, bundle_path="model_bundle.pkl"):
    # bundle 로드
    bundle = joblib.load(bundle_path)

    preprocessor = bundle['preprocessor']
    model = bundle['model']
    threshold = bundle['threshold']
    feature_names = bundle['feature_columns']

    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    df = df.drop(columns=["customerID"], errors="ignore")

    # 컬럼 정렬
    df=df.reindex(columns=feature_names, fill_value=0)

    # 전처리 적용
    X_processed = preprocessor.transform(df)

    # 예측
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities

if __name__ == "__main__":
    file_path = "test_data.csv"  # 예측할 CSV 파일 경로
    predictions, probabilities = predict_from_csv(file_path)
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
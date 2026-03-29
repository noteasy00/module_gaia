import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load("model.pkl")
preprocessor = joblib.load("prepocessor.pkl")

# 전처리 저장했을 경우
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)

    X = df.drop('customerID', axis=1, errors='ignore')

    X_processed = preprocessor.transform(X)

    predictions = model.predict(X_processed)

    df['prediction'] = predictions

    return df


# 전처리 필요한 경
def predic_from_csv(file_path):
    df = pd.read_csv(file_path)

    X=df.copy()
    X = X.drop('customerID', axis=1, errors='ignore')

    # 이진형 데이터 변환 (Label Encoding)
    binary_cols = ['PaperlessBilling']

    le = LabelEncoder()
    for col in binary_cols:
        X[col] = le.fit_transform(X[col])

    # 다중 클래스 범주 데이터 변환 (One-Hot Encoding)
    multi_cols = ['InternetService',
                   'TechSupport',
                   'StreamingMovies',
                  'Contract', 
                  'PaymentMethod'
    ]
    X = pd.get_dummies(X, columns=multi_cols, drop_first=True)

    # 컬럼 맞추기
    model_colums = model. feature_names_in_
    X = X.reindex(columns = model_colums, fill_value=0)

    predictions = model.predict(X)

    df['prediction'] = predictions

    return df

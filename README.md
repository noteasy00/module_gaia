# GAIA Churn Defense System

통신사 고객 이탈 예측 및 방어 전략 제안 시스템입니다.

## 주요 기능
- 고객 CSV 업로드 후 이탈 확률 예측
- 고위험 고객 선별
- 할인/혜택 적용 시 이탈 위험도 시뮬레이션
- 고위험 고객 대상 마케팅 전략 및 메시지 제안
- 모델 성능 지표 확인 (Accuracy, F1, Precision, Recall, PR-AUC, ROC-AUC, Confusion Matrix)

## 프로젝트 구조
- `src/model/` : 모델 학습 코드
- `src/app/` : FastAPI / Streamlit 프론트
- `src/agent/` : 전략 제안 에이전트
- `ckpt/` : 학습된 모델 번들 저장
- `outputs/` : 성능 지표 및 confusion matrix 저장

## 사전 준비
1. 가상환경 생성 및 활성화
2. 패키지 설치
3. `.env` 파일 생성 후 OpenAI API Key 입력

## 설치
```bash
pip install -r requirements.txt

## 모델 학습
python src/model/Pasted\ code.py

실행 후 아래 파일이 생성되어야 합니다:
ckpt/xgb_top5_cv.pkl
outputs/xgb_top5_cv_test_confusion_matrix.png

## FastAPI 서버 실행
app.py 실제 위치에 따라 아래 중 맞는 명령 사용
프로젝트 루트(module_gaia)에서 실행 시, 
uvicorn src.app.app:app --reload --host 127.0.0.1 --port 8000
uvicorn src.agent.app:app --reload --host 127.0.0.1 --port 8000 위에 안되면 이걸로

## Streamlit 프론트 실행
python -m streamlit run src/app/front.py

## Agent 실행
python src/agent/agent.py

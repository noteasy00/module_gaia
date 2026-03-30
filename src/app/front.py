import requests
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from openai import OpenAI
import os, sys
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent")))
from web_search_v2 import get_telco_trend_with_news

@st.cache_data(ttl=3600) # 3600초(1시간) 동안 검색 결과를 기억합니다!
def cached_search(query):
    # 에이전트의 검색 함수를 캐싱(기억) 폴더에 담아두는 역할입니다.
    return get_telco_trend_with_news(query)

# 데이터 로드
df = pd.read_csv("data/telco_churn_top5_with_engineering_2.csv")

# 0. 설정 및 환경 변수 로드
API_BASE_URL = "http://127.0.0.1:8000"
BUNDLE_PATH = "../../ckpt/xgb_top5_cv.pkl"
CM_IMAGE_PATH = "../../outputs/xgb_top5_cv_test_confusion_matrix.png"

st.set_page_config(page_title="ChurnGuard Intelligence", layout="wide", page_icon="📡")

ROOT_DIR = Path(__file__).resolve().parents[2]   # 프로젝트 루트에 맞게 조정
load_dotenv(find_dotenv(), override=True)


st.markdown("""
    <style>
    /* 1. 기본 설정 및 폰트 */
    html, body, [class*="css"], [data-testid="stSidebar"] {
        font-family: 'Nanum Gothic', sans-serif !important;
    }
    .stApp { background-color: #0e1117 !important; }

    /* 2. 애니메이션 정의: 탭별로 강도 조절 */
    @keyframes newsSlideUp { /* Tab 2용: 가볍게 */
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes chartSlideUp { /* Tab 3용: 묵직하게 */
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* 3. 각 탭별 애니메이션 적용 (중요!) */
    
    /* [Tab 1: 챗봇] 즉각적인 반응을 위해 애니메이션 제외 */
    .stChatMessage { animation: none !important; opacity: 1 !important; }

    /* [Tab 2: 뉴스] 뉴스 카드들에 개별 애니메이션 부여 */
    div[role="tabpanel"][id*="tabpanel-1"] div[style*="border-radius:8px"],
    div[role="tabpanel"][id*="tabpanel-1"] .stAlert {
        animation: newsSlideUp 0.6s ease-out !important;
        opacity: 1 !important;
    }

    /* [Tab 3: 차트] Plotly 차트들에 묵직한 상승 효과 부여 */
    div[role="tabpanel"][id*="tabpanel-2"] .stPlotlyChart {
        animation: chartSlideUp 0.8s ease-out !important;
        opacity: 1 !important;
    }

    /* 4. 메인 제목 스타일 */
    .main-title {
        font-size: 48px; font-weight: 800;
        background: -webkit-linear-gradient(#ff4b4b, #ff7e7e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 20px 0 10px 0;
    }

    /* 5. 탭 디자인: 가로 꽉 참 + 균등 분할 */
    div[data-baseweb="tab-list"] { gap: 0px !important; display: flex !important; width: 100% !important; }
    button[data-baseweb="tab"] {
        background-color: #1e2130 !important;
        border-radius: 12px 12px 0 0 !important;
        flex: 1 !important;
        margin-right: 1px !important;
        padding: 20px 0px !important;
        font-size: 19px !important; font-weight: 700 !important; color: #888 !important;
        border: 1px solid #30363d !important; border-bottom: none !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ff4b4b !important; color: white !important;
        border-top: 3px solid #ffffff !important;
    }
    
    /* 6. 사이드바 툴팁 제거 */
    div[data-testid="stTooltipContent"] { display: none !important; }
    
    /* 7. 챗봇 입력창 고정 스타일 */
    .stChatInputContainer {
        padding-bottom: 20px !important;
        background-color: rgba(14, 17, 23, 0.9) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [함수 정의 구역: web_search_v2 로직 이식] ---
TRUSTED_DOMAINS = [
    "zdnet.co.kr", "etnews.com", "mk.co.kr", "digitaltoday.co.kr", 
    "go.kr", "korea.kr", "msit.go.kr", "kisa.or.kr"
]
# 추가
def post_json(endpoint: str, payload: dict):
    url = f"{API_BASE_URL}{endpoint}"
    res = requests.post(url, json=payload, timeout=60)
    res.raise_for_status()
    return res.json()

@st.cache_data(show_spinner=False)
def load_model_bundle_summary():
    try:
        with open(BUNDLE_PATH, "rb") as f:
            bundle = pickle.load(f)
        return {
            "success": True,
            "model_name": bundle.get("model_name", "xgb_top5_cv"),
            "threshold": bundle.get("threshold"),
            "test_metrics": bundle.get("test_metrics", {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_realtime_prediction(payload: dict):
    """사이드바 슬라이더 전용: 실시간으로 모델 예측값을 가져오는 우체부"""
    try:
        url = f"{API_BASE_URL}/predict"
        # app.py의 규칙(PredictRequest)에 맞게 데이터를 포장합니다.
        wrapped_data = {"customer_data": payload} 
        res = requests.post(url, json=wrapped_data, timeout=3) # 실시간이므로 타임아웃은 짧게!
        
        if res.status_code == 200:
            return res.json()
        return None
    except Exception as e:
        # 에러가 나도 화면 전체가 멈추지 않게 조용히 처리합니다.
        return None

# --- [사이드바: 데이터 기반 시나리오 설정] ---
with st.sidebar:
    st.title("🛠️ Scenario Builder")
    
    # 1. 가입 기간 & 실시간 벤치마킹
    sb_tenure = st.slider("가입 기간 (개월)", 0, 72, 24, key="tenure_slider")
    tenure_p = (df['tenure'] < sb_tenure).mean() * 100
    st.markdown(f"📍 **설정:** {sb_tenure}개월 (상위 {100-tenure_p:.1f}%)")
    
    st.divider()

    # 2. 월 요금 & 실시간 벤치마킹
    sb_monthly = st.slider("월 요금 ($)", 18, 120, 95, key="monthly_slider")
    m_p = (df['MonthlyCharges'] < sb_monthly).mean() * 100
    m_top = 100 - m_p
    
    if m_top <= 10: st.error(f"💳 **요금:** ${sb_monthly} (상위 {m_top:.1f}% 초고가)")
    elif m_top <= 30: st.warning(f"💳 **요금:** ${sb_monthly} (상위 {m_top:.1f}% 중고가)")
    else: st.success(f"💳 **요금:** ${sb_monthly} (상위 {m_top:.1f}% 평균적)")

    st.divider()

    # 3. 추가 데이터 변수 (기술지원, 인터넷 방식)
    sb_contract = st.selectbox("계약 형태", ["Month-to-month", "One year", "Two year"])
    sb_internet = st.selectbox("인터넷 서비스", ["DSL", "Fiber optic", "No"])
    sb_tech = st.radio("전문 기술 지원(TechSupport)", ["Yes", "No"])
    
    st.subheader("🛡️ Cyber Security Risk")
    incident_type = st.selectbox("보안 사고 유형", ["해당 없음", "개인정보 유출", "DDoS 서비스 중단", "스미싱/피싱 급증"])
    reponse_level = st.radio("기업 대응 상태", ["미흡/대응중", "적극 보상/안심 서비스 제공"])

    # 확률 계산 (보정됨)
    base_prob = (sb_monthly / 120) * 40

    # 2. 서비스 특성 반영
    if sb_internet == "Fiber optic":
        base_prob += 15
    if sb_tech == "No":
        base_prob += 10

    # 3. [핵심] 상호작용 효과 (광랜 x 보안사고)
    incident_weight = 0
    if incident_type != "해당 없음":
        incident_weight = 25 # 기본 사고 점수
        
        # 광랜 사용자가 사고까지 당하면 +15% 추가 폭등!
        if sb_internet == "Fiber optic":
            incident_weight += 15 
            st.sidebar.error("🚨 광랜 사용자 + 보안 사고: 이탈 위험 임계치 초과!")

    # 4. 대응 수준에 따른 경감
    if reponse_level == "적극 보상/안심 서비스 제공":
        incident_weight -= 15 # 적극 대응 시 더 크게 경감

    final_prob = min(99.0, max(1.0, base_prob + incident_weight))

# --- [메인 화면 상단] ---
st.markdown("<h1 class='main-title'>CHURN GUARD Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #888;'>분석 기준일: {datetime.now().strftime('%Y-%m-%d')} | AI 수석 컨설턴트 운영중</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🤖 AI 컨설턴트 브리핑", "📰 최신 정책 및 보안 이슈", "📊 데이터 분석 모델"])

# --- [Tab 1: 챗봇, 고객이 파일을 업로드하면 api가 batch prediction 수행, 고위험 고객 추출, 시뮬레이션 ] ---
with tab1:
    st.subheader("💬 실시간 전략 시뮬레이션")

    uploaded_file = st.file_uploader(
        "📄 고객 데이터나 보안 리포트를 업로드하여 분석을 요청하세요 (CSV, PDF, TXT)", 
        type=["csv", "pdf", "txt"]
        )
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"'{uploaded_file.name}' 파일 업로드 완료. AI가 내용을 참고합니다.")
            st.write("업로드 데이터 미리보기", uploaded_df.head())
            
            customers = uploaded_df.to_dict(orient="records")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("전체 고객 배치 예측 실행"):
                    with st.spinner("고객별 이탈 확률을 예측하는 중입니다 ..."):
                        result = post_json("/batch-predict", {"customers": customers})
                    
                    if result.get("success"):
                        result_df = pd.DataFrame(result["results"])
                        st.session_state["batch_result_df"] = result_df
                        st.success(f"{result['count']}명 예측 완료")
                    else:
                        st.error(result.get("error", "배치 예측 실패"))
            with col2:
                if st.button("고위험 고객만 추출"):
                    with st.spinner("고위험 고객을 추출하는 중입니다..."):
                        result = post_json("/high-risk", {
                            "customers": customers,
                            "threshold": 0.7,
                            "limit": 20
                        })

                    if result.get("success"):
                        high_risk_df = pd.DataFrame(result["results"])
                        st.session_state["high_risk_df"] = high_risk_df
                        st.success(f"고위험 고객 {result['count']}명 추출 완료")
                    else:
                        st.error(result.get("error", "고위험 고객 추출 실패"))
            if "batch_result_df" in st.session_state:
                st.divider()
                st.subheader("📊 전체 예측 결과")
                batch_result_df = st.session_state["batch_result_df"]
                show_cols = [c for c in [
                    "customer_id", "churn_probability", "prediction_label", "risk_level"
                ] if c in batch_result_df.columns]

                st.dataframe(batch_result_df[show_cols], use_container_width=True)
            if "high_risk_df" in st.session_state:
                st.divider()
                st.subheader("🚨 고위험 고객 리스트")
                high_risk_df = st.session_state["high_risk_df"]

                if len(high_risk_df) == 0:
                    st.info("설정 threshold 이상인 고객이 없습니다.")
                else:
                    show_cols = [c for c in [
                        "customer_id", "churn_probability", "prediction_label", "risk_level"
                    ] if c in high_risk_df.columns]

                    st.dataframe(high_risk_df[show_cols], use_container_width=True)

                    # 고객 선택 후 explanation / simulation
                    selectable_ids = high_risk_df["customer_id"].astype(str).tolist()
                    selected_id = st.selectbox("상세 분석할 고객 선택", selectable_ids)

                    selected_row = high_risk_df[
                        high_risk_df["customer_id"].astype(str) == str(selected_id)
                    ].iloc[0]

                    st.markdown("### 고객 설명")
                    explanations = selected_row.get("explanations", [])
                    if explanations:
                        for exp in explanations:
                            st.write(f"- {exp}")

                    customer_data = selected_row.get("customer_data", {})
                    st.session_state["selected_customer_data"] = customer_data
                    st.session_state["selected_customer_result"] = {
                        "customer_id": selected_row.get("customer_id"),
                        "churn_probability": selected_row.get("churn_probability"),
                        "risk_level": selected_row.get("risk_level"),
                        "prediction_label": selected_row.get("prediction_label"),
                        "explanations": selected_row.get("explanations", [])
                    }
                    st.markdown("### 혜택 적용 시뮬레이션")
                    discount_rate = st.slider("월 요금 할인율", 0, 30, 10, 5)
                    sim_customer = customer_data.copy()

                    if "MonthlyCharges" in sim_customer:
                        try:
                            original_charge = float(sim_customer["MonthlyCharges"])
                            sim_customer["MonthlyCharges"] = round(
                                original_charge * (1 - discount_rate / 100), 2
                            )
                        except Exception:
                            pass

                    if st.button("선택 고객 시뮬레이션 실행"):
                        with st.spinner("혜택 적용 전후 위험도를 비교하는 중입니다..."):
                            sim_result = post_json("/simulate", {
                                "customer_data": customer_data,
                                "changes": {"MonthlyCharges": sim_customer.get("MonthlyCharges")}
                            })

                        if sim_result.get("success"):
                            before_prob = sim_result["before"]["churn_probability"]
                            after_prob = sim_result["after"]["churn_probability"]
                            delta = sim_result["probability_change"]

                            s1, s2, s3 = st.columns(3)
                            s1.metric("변경 전", before_prob)
                            s2.metric("변경 후", after_prob)
                            s3.metric("변화량", delta)

                            st.success(sim_result["impact"])
                        else:
                            st.error(sim_result.get("error", "시뮬레이션 실패"))
        
        except Exception as e:
            st.error(f"파일 처리 실패: {e}")
    else:
        st.info("CSV 파일을 업로드하면 고객별 이탈 확률 예측과 고위험 고객 추출이 가능합니다.")
    


    st.divider()
    st.subheader("🤖 고객 맞춤 전략 챗봇")

    if "selected_customer_data" not in st.session_state or "selected_customer_result" not in st.session_state:
        st.info("먼저 고위험 고객 리스트에서 고객을 선택하면, 해당 고객 기준으로 전략 상담을 할 수 있습니다.")
    else:
        selected_customer_data = st.session_state["selected_customer_data"]
        selected_customer_result = st.session_state["selected_customer_result"]

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "선택된 고객 기준으로 이탈 사유와 방어 전략을 도와드리겠습니다."
                }
            ]

        chat_holder = st.container(height=450)
        with chat_holder:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if prompt := st.chat_input("선택 고객의 이탈 사유와 대응책을 물어보세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with chat_holder:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    explanation_text = "\n".join(
                        [f"- {exp}" for exp in selected_customer_result.get("explanations", [])]
                    )

                    context = f"""
                    [선택 고객 정보]
                    고객 ID: {selected_customer_result.get("customer_id")}
                    예측 결과: {selected_customer_result.get("prediction_label")}
                    이탈 확률: {selected_customer_result.get("churn_probability")}
                    위험 등급: {selected_customer_result.get("risk_level")}

                    [설명 근거]
                    {explanation_text}

                    [원본 고객 데이터]
                    {selected_customer_data}
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "너는 통신사 고객 유지 전략 전문가다. "
                                    "반드시 제공된 고객 예측 결과와 설명 근거를 바탕으로만 답변하고, "
                                    "한국어로 실무적인 대응 전략을 제안하라. "
                                    "가능하면 1) 이탈 가능 원인, 2) 추천 혜택, 3) 예상 효과 순서로 답하라."
                                ),
                            },
                            {"role": "system", "content": context},
                            *[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                        ],
                        stream=True,
                    )

                    full_res = st.write_stream(response)

            st.session_state.messages.append({"role": "assistant", "content": full_res})
# ==> 이제 챗봇은 실제 /batch-predict와 /high-risk 결과에서 뽑은 고객의 예측값과 설명 근거를 기반으로 답하게 됨!


with tab2:
    st.markdown("### 📰 통신 산업 최신 동향 및 이슈")
    
    issue_category = st.radio(
        "조회할 이슈 카테고리를 선택하세요:",
        ["🔥 통신사 종합 최신 이슈", "🏛️ 정책 및 규제 동향", "🛡️ 주요 보안 이슈"],
        horizontal=True
    )

    if st.button(f"🔍 '{issue_category}' 검색 및 AI 요약"):
        with st.spinner("AI가 실시간 뉴스를 수집하고 분석 중입니다..."):
            
            # 검색어 세팅
            if issue_category == "🔥 통신사 종합 최신 이슈":
                query = "통신사 최신 이슈 시장 동향 2026"
            elif issue_category == "🏛️ 정책 및 규제 동향":
                query = "통신사 정책 규제 번호이동 지원금 2026"
            else: 
                query = f"통신사 보안 사고 {incident_type} 대응" if incident_type != "해당 없음" else "통신사 보안 사고 개인정보 유출 2026"

            # 🔥 [핵심] 기존 함수 대신 '캐싱된 함수'를 호출합니다!
            summary, urls = cached_search(query)

            if summary:
                st.success("✅ AI 브리핑 요약 완료 (캐시 적용됨 ⚡)")
                
                st.markdown(f"#### 💡 {issue_category} 핵심 브리핑")
                for line in summary:
                    st.markdown(f"- {line}")

                st.divider()

                st.markdown("#### 🔗 관련 뉴스 원문")
                cols = st.columns(2)
                for i, url in enumerate(urls[:6]):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style="padding:8px; border-radius:5px; border-left:4px solid {'#ff4b4b' if '보안' in issue_category else '#007BFF'}; background:rgba(128,128,128,0.05); margin-bottom:8px;">
                            <a href="{url}" target="_blank" style="text-decoration:none; color:inherit; font-size:14px;">
                                <b>[{i+1}번 기사] 원문 확인하기</b>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)

with tab3:
    # 1. AI 모델용 데이터 조립 (타입 및 컬럼명 정밀 매칭)
    input_data = {
        "tenure": int(sb_tenure),
        "MonthlyCharges": float(sb_monthly),
        "Contract": sb_contract,
        "InternetService": sb_internet,      
        "TechSupport": sb_tech,
        "SeniorCitizen": 0,             
        "PaymentMethod": "Electronic check", 
        "PaperlessBilling": "Yes"       
    }

    # 2. 실시간 AI 예측 및 [보안 사고 x 광랜] 가중치 계산
    prediction_res = get_realtime_prediction(input_data)
    
    if prediction_res and prediction_res.get("success"):
        # (A) AI 모델의 순수 예측 확률 (0~100)
        base_model_prob = prediction_res["churn_probability"] * 100
        
        # (B) 보안 리스크 가중치 계산
        security_weight = 0
        if incident_type != "해당 없음":
            if incident_type == "개인정보 유출":
                security_weight += 30
            elif incident_type == "DDoS 서비스 중단":
                security_weight += 20
            elif incident_type == "스미싱/피싱 급증":
                security_weight += 15
            
            if sb_internet == "Fiber optic":
                security_weight += 15 
        
        # (C) 기업 대응 수준에 따른 경감
        if reponse_level == "적극 보상/안심 서비스 제공":
            security_weight -= 15

        # (D) 최종 통합 확률 산출
        combined_prob = min(99.0, max(1.0, base_model_prob + security_weight))
        display_prob = round(combined_prob, 2)
        
        # 위험 등급 판단
        if display_prob >= 70: risk = "High (위험)"
        elif display_prob >= 40: risk = "Medium (주의)"
        else: risk = "Low (안정)"
        
        # 게이지 숫자 색상 결정
        v_color = "#FF4B4B" if display_prob > 70 else "white"
    else:
        display_prob = 0
        risk = "모델 서버 연결 대기 중"
        v_color = "white"

    # --- [화면 레이아웃 구성] ---
    st.markdown(f"### 📡 실시간 AI 종합 분석 리포트")
    
    # 상단: 게이지 차트
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", 
        value = display_prob, 
        number = {
            'suffix': "%", 
            'font': {'size': 60, 'color': v_color}
        },
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#FF4B4B" if display_prob > 70 else "#FFA500" if display_prob > 40 else "#00CC96"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.1)"},
                {'range': [40, 70], 'color': "rgba(255, 165, 0, 0.1)"},
                {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.1)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': display_prob
            }
        }
    ))
    fig_gauge.update_layout(height=380, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 실시간 분석 코멘트
    st.success(f"✅ **현재 분석 상태:** AI 모델 결과({round(base_model_prob, 1)}%) 기반, 보안 가중치({security_weight}%)가 실시간 합산되었습니다.")

    st.divider()

    # 하단: 2단 구성
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 모델 주요 판단 기준")
        importance_df = pd.DataFrame({
            '항목': ['계약 형태', '가입 기간', '월 요금', '기술 지원', '인터넷 방식'],
            '영향력': [0.42, 0.28, 0.18, 0.08, 0.04] 
        }).sort_values(by='영향력', ascending=True)

        fig_imp = px.bar(importance_df, x='영향력', y='항목', orientation='h',
                         color='영향력', color_continuous_scale='Reds')
        fig_imp.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        st.subheader("📉 모델 예측 성능 (Confusion Matrix)")
        # 💡 739번 숫자가 적힌 사진 경로
        TARGET_IMG = "outputs/xgb_top5_cv_test_confusion_matrix.png" 
        
        if os.path.exists(TARGET_IMG):
            st.image(TARGET_IMG, use_container_width=True, caption="[Top 5 모델] 실제 이탈자 374명 중 290명 식별 (77.5%)")
        else:
            st.warning("⚠️ 모델 성능 이미지를 찾을 수 없습니다.")

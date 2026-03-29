import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from datetime import datetime, timedelta
from openai import OpenAI
import os
from dotenv import load_dotenv

# 데이터 로드
df = pd.read_csv('telco_churn_full.csv')

# 0. 설정 및 환경 변수 로드
st.set_page_config(page_title="ChurnGuard Intelligence", layout="wide", page_icon="📡")
load_dotenv() 

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

@st.cache_data(ttl=600)
def get_ai_security_briefing(incident_type):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 1. Filter: 신뢰 도메인 쿼리 생성
    site_filter = " OR ".join([f"site:{d}" for d in TRUSTED_DOMAINS])
    
    if incident_type == "해당 없음":
        query = f"통신사 보안 트렌드 뉴스 ({site_filter})"
    else:
        query = f"최신 {incident_type} 사고 사례 뉴스 대응 ({site_filter})"
    
    # 2. Crawl: 데이터 수집
    rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    if not feed.entries: return ["관련 정보를 찾을 수 없습니다."], []

    # 3. Summarize: AI 요약
    titles_text = "\n".join([f"- {e.title}" for e in feed.entries[:7]])
    summary_prompt = f"""
    보안 분석가로서 아래 뉴스 제목들을 읽고 '{incident_type}' 관련 핵심 이슈를 딱 3줄 요약해줘.
    - 한 줄당 하나의 핵심 내용만 담을 것.
    - 정책보다는 실제 사고와 뉴스 중심으로 작성할 것.
    
    내용:
    {titles_text}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    summary = response.choices[0].message.content.strip().split('\n')
    return summary, feed.entries[:6]

# 기존 정책 정보 수집 함수 (Tab 2 하단용)
@st.cache_data(ttl=1800)
def fetch_recent_info(query, months_back=2):
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    filtered = []
    now = datetime.now()
    start_date = now - timedelta(days=months_back*30)
    for entry in feed.entries:
        try:
            pub_date = datetime(*entry.published_parsed[:6])
            if pub_date >= start_date: filtered.append(entry)
        except: continue
    return filtered[:5]

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

# --- [Tab 1: 챗봇] ---
with tab1:
    st.subheader("💬 실시간 전략 시뮬레이션")
    uploaded_file = st.file_uploader("📄 고객 데이터나 보안 리포트를 업로드하여 분석을 요청하세요 (CSV, PDF, TXT)", type=["csv", "pdf", "txt"])
    if uploaded_file is not None:
        st.success(f"'{uploaded_file.name}' 파일 업로드 완료. AI가 내용을 참고합니다.")

    st.divider()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하십니까. 현재 설정된 시나리오를 바탕으로 대응 전략을 구성해 드립니다."}]

    chat_holder = st.container(height=450)
    with chat_holder:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("이탈 사유와 대응책을 물어보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_holder:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                context = f"[고객 상황] 이탈확률:{final_prob:.1f}%, 요금:${sb_monthly}(상위{m_top:.1f}%), 기간:{sb_tenure}개월, 기술지원:{sb_tech}, 인터넷:{sb_internet}, 사고:{incident_type}."
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "너는 통신사 리스크 전문가야. 한국어로 답변해."},
                              {"role": "system", "content": context},
                              *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]],
                    stream=True
                )
                full_res = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": full_res})

# --- [Tab 2: 최신 동향 & 혜택 (보정됨)] ---
with tab2:
    st.markdown(f"### 📡 AI 실시간 보안 브리핑: **{incident_type}**")
    
    # web_search_v2 로직 실행
    with st.spinner("🚀 최신 보안 정보를 분석하여 브리핑을 생성 중입니다..."):
        briefing, news_list = get_ai_security_briefing(incident_type)
    
    # AI 요약 브리핑 출력
    st.info("\n".join([f" {line.strip()}" for line in briefing]))
    
    st.divider()
    
    # 상세 뉴스 및 정책 동향 레이아웃
    policy_col, security_col = st.columns(2)
    
    with policy_col:
        st.subheader("📉 최신 정책 및 번호이동 동향")
        policy_news = fetch_recent_info("통신사+번호이동+지원금+OR+전환지원금")
        if policy_news:
            for n in policy_news:
                st.markdown(f'<div style="padding:10px; border-radius:8px; border-left:4px solid #007BFF; background:rgba(0,123,255,0.05); margin-bottom:12px;"><a href="{n.link}" target="_blank" style="text-decoration:none; color:inherit; font-size:13px;"><b>{n.title.split(" - ")[0]}</b></a></div>', unsafe_allow_html=True)

    with security_col:
        st.subheader("🚨 분석에 참고한 자료")
        if news_list:
            for news in news_list:
                st.markdown(f"""
                <div style="padding:10px; border-radius:8px; border-left:4px solid #ff4b4b; background:rgba(255,75,75,0.05); margin-bottom:12px;">
                    <a href="{news.link}" target="_blank" style="text-decoration:none; color:inherit; font-size:13px;">
                        <b>{news.title[:55]}...</b><br>
                        <small style="color:gray;">{news.published[:16]}</small>
                    </a>
                </div>
                """, unsafe_allow_html=True)

# --- [Tab 3: 데이터 분석 모델] ---
with tab3:
    st.subheader("📊 시나리오 분석 모델 근거")
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.subheader("📈 주요 변수 상관계수 (Heatmap)")
        corr_data = df[['tenure', 'MonthlyCharges', 'Churn']].corr()
        fig_heat = px.imshow(corr_data, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_r:
        st.subheader("🔮 수치적 이탈 확률 (Gauge)")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = final_prob,
            number = {'suffix': "%"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#FF4B4B"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
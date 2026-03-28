import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# 최신 동향 서치

# 1. 공신력 있는 사이트 한정
# -------------------------
ALL_DOMAINS = [
    # 공공 / 연구
    "go.kr",
    "or.kr",
    "ac.kr",
    "korea.kr",
    "msit.go.kr",
    "kisdi.re.kr",
    "nia.or.kr",

    # 뉴스
    "chosun.com",
    "joongang.co.kr",
    "donga.com",
    "hani.co.kr",
    "mk.co.kr",
    "mt.co.kr",
    "edaily.co.kr",
    "etnews.com",
    "zdnet.co.kr",
    "electimes.com",
    "digitaltoday.co.kr"
]

def web_search_trusted(query):
    site_filter = " OR ".join([f"site:{d}" for d in ALL_DOMAINS])
    search_query = f"{query} ({site_filter})"

    response = client.responses.create(
        model = "gpt-5.4",
        tools=[{"type": "web_search"}],
        input=search_query
    )

    return response.output_text


# 2. URL 추출
# -------------------------
def extract_urls(text, k=10):
    urls = re.findall(r'https?://[^\s)]+', text)
    urls = list(set(urls))
    return urls[:k]


# 3. 요약 
# -------------------------
def summarize_results(web_text, max_lines=10):

    summary_prompt = f"""
    아래 내용을 기반으로 통신 산업 최신 동향을 {max_lines}줄로 정리하라.
    반드시:
    - 한 줄당 하나의 핵심 내용
    - 짧고 명확하게
    - bullet 없이 텍스트만

    내용:
    {web_text}
    """

    response = client.responses.create(
        model="gpt-5.4",
        input=summary_prompt
    )

    lines = [line.strip() for line in response.output_text.split("\n") if line.strip()]

    return lines[:max_lines]

# 4. 최종 함수
# -------------------------
def get_telco_trend_with_news(query="통신 산업 최신 동향 2026", k=10):
    web_text = web_search_trusted(query)

    summary = summarize_results(web_text, max_lines=k)
    urls = extract_urls(web_text, k=k)

    return summary, urls

# -------------------------
# 실행
# -------------------------

if __name__ == "__main__":

    summary, urls = get_telco_trend_with_news()

    print("통신 산업 최신 동향 (요약)")
    for line in summary:
        print("-", line)

    print("\n관련 뉴스 URL")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")

# 라우터
def is_telco_trend_query(user_input):
    telco_keywords = ["통신", "통신사", "5G", "인터넷", "telecom"]
    trend_keywords = ["동향", "현황", "트렌드", "뉴스", "최근", "최신"]

    return (
        any(k in user_input for k in telco_keywords) and
        any(t in user_input for t in trend_keywords)
    )

def route_user_query(user_input):

    if is_telco_trend_query(user_input):
        summary, urls = get_telco_trend_with_news(user_input)
        return summary, urls

    # 기본 LLM 응답
    #return general_chat(user_input)
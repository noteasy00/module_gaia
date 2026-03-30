import os
import sys
import time
import json
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
tools_folder = os.path.abspath(os.path.join(current_dir, '..', 'tools'))
sys.path.append(tools_folder)
from src.tools.web_search_v2 import get_telco_trend_with_news

ROOT_DIR = Path(__file__).resolve().parents[2]   # 프로젝트 루트에 맞게 조정
load_dotenv(ROOT_DIR / ".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================
# 1. AI 에이전트 프롬프트
# ==========================================
SYSTEM_INSTRUCTIONS = """
# 페르소나
너는 통신사 고객 유지 전략을 담당하는 '마케팅 컨설턴트'이자 '데이터 분석가'야. 
머신러닝 모델이 예측한 이탈 위험 수치를 바탕으로, 회사의 수익을 극대화하고 이탈을 최소화하는 최적의 방어 전략을 세워서 사용자에게 제안해야 해.

# 작업 지침
1. 데이터 분석 및 타겟팅:
   - 입력된 데이터에서 이탈 확률이 높은 고객을 우선적으로 식별한다.
   - 특히 월평균 요금(Monthly Charges)이 높고 가입 기간(Tenure Months)이 긴 '고가치(High-Value)' 고객을 최우선 방어 대상으로 선정한다.

2. 개인화된 방어 전략 수립:
   - 고객별 특성(서비스 이용 패턴, 요금제 등)에 맞춰 '왜 이 사람이 떠나려 하는가'를 가설을 세우고 그에 맞는 혜택을 결정한다.
   - 예: 요금이 부담스러운 고객 -> 요금 할인 프로모션 / 혜택이 부족한 고객 -> 부가 서비스(OTT 등) 무료 체험권.

3. 실행 문구 생성:
   - 선정된 고객에게 보낼 마케팅 메시지를 고객별로 작성한다.
   - 메시지는 짧고 설득력 있게 작성한다.

4. 툴 사용 원칙:
    - 서버 연결 상태를 확인해야 할 경우 check_api_health 도구를 사용한다.
    - 여러 고객을 분석할 때는 predict_batch_customers를 우선 사용한다.
    - 고위험 고객 선별이 필요하면 get_high_risk_customers를 사용한다.
    - 혜택 전후 효과 비교가 필요하면 simulate_customer_offer를 사용한다.

# 출력 가이드
결과물은 반드시 다음 구조를 포함해야 한다:
- [타겟 요약]: 선정된 핵심 타겟 고객 수와 그들의 공통적 특징.
- [방어 전략]: 적용할 프로모션의 종류와 선정 사유 (비용 대비 효율성 고려).
- [메시지 리스트]: 고객 ID별로 발송할 실제 마케팅 문구 예시.
- [기대 효과]: 해당 액션 실행 시 예상되는 손실 방어 금액(ROI) 추정치.

# 제약 사항
- 비즈니스 용어를 사용하되, 실행 가능한(Actionable) 제안을 할 것.
- 답변은 반드시 한국어로, 너무 길지 않게, 발표용 데모에 적합한 형태로 정리할 것.
- 근거 없는 수치를 임의로 만들지 않는다.
- ROI는 명확한 비용 정보가 없으면 정성적 효과 또는 가정 기반 추정으로 제시한다.
"""

# ==========================================
# 2. Tools & Functions 정의
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_single_customer",
            "description": "고객 한 명의 데이터를 바탕으로 이탈 확률을 예측합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_data": {"type": "object", "description": "고객 데이터 딕셔너리"}
                },
                "required": ["customer_data"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "check_api_health",
        "description": "FastAPI 서버가 정상적으로 동작하는지 확인합니다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_customer_offer",
            "description": "고객에게 할인 등 조건을 변경했을 때 이탈 확률이 어떻게 변하는지 시뮬레이션합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_data": {"type": "object", "description": "기존 고객 데이터"},
                    "changes": {"type": "object", "description": "변경할 조건 (예: {'Monthly Charges': 50000})"}
                },
                "required": ["customer_data", "changes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_batch_customers",
            "description": "여러 명의 고객 리스트를 한 번에 입력받아 일괄 예측합니다. 필수: customers 배열 데이터.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customers": {"type": "array", "items": {"type": "object"}, "description": "고객 데이터 리스트"}
                },
                "required": ["customers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_high_risk_customers",
            "description": "전체 고객 리스트 중 이탈 위험(threshold)이 높은 고객만 필터링합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customers": {"type": "array", "items": {"type": "object"}, "description": "전체 고객 리스트"},
                    "threshold": {"type": "number", "description": "이탈 위험 기준선 (예: 0.7)"},
                    "limit": {"type": "integer", "description": "가져올 최대 고객 수 (선택사항)"}
                },
                "required": ["customers", "threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_promotions",
            "description": "선별된 타겟 고객들에게 마케팅 프로모션 메시지를 발송(시뮬레이션)합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "프로모션을 발송할 대상 고객 ID 리스트"
                    },
                    "discount_rate": {
                        "type": "number",
                        "description": "적용할 요금 할인율 (예: 0.1은 10%)"
                    }
                },
                "required": ["target_ids", "discount_rate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_telco_trends",
            "description": "통신 산업의 최신 동향, 뉴스, 보안 이슈 등을 인터넷에서 검색하고 요약합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "검색어 (예: '통신 산업 최신 동향 2026')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ==========================================
# 3. 에이전트 설정
# ==========================================
def setup_agent():
    assistant = client.beta.assistants.create(
        name="Churn Defense Marketing Agent",
        instructions=SYSTEM_INSTRUCTIONS,
        model="gpt-4o",
        tools=tools
    )
    return assistant

# ==========================================
# 4. FastAPI 통신 및 툴 실행 로직
# ==========================================
API_BASE_URL = "http://127.0.0.1:8000"

def execute_ml_logic(tool_calls):
    tool_outputs = []

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"\n[시스템] AI가 {func_name} 기능을 실행합니다...")

        output = {}
        try:
            if func_name == "predict_single_customer":
                res = requests.post(f"{API_BASE_URL}/predict", json=args, timeout=30)
                res.raise_for_status()
                output = res.json()

            elif func_name == "simulate_customer_offer":
                res = requests.post(f"{API_BASE_URL}/simulate", json=args, timeout=30)
                res.raise_for_status()
                output = res.json()

            elif func_name == "predict_batch_customers":
                res = requests.post(f"{API_BASE_URL}/batch-predict", json=args, timeout=60)
                res.raise_for_status()
                output = res.json()

            elif func_name == "get_high_risk_customers":
                res = requests.post(f"{API_BASE_URL}/high-risk", json=args, timeout=60)
                res.raise_for_status()
                output = res.json()

            elif func_name == "check_api_health":
                res = requests.get(f"{API_BASE_URL}/health", timeout=10)
                res.raise_for_status()
                output = res.json()

            elif func_name == "send_promotions":
                target_ids = args.get("target_ids", [])
                discount_rate = args.get("discount_rate", 0)

                output = {
                    "success": True,
                    "sent_count": len(target_ids),
                    "discount_rate": discount_rate,
                    "message": f"{len(target_ids)}명의 고객에게 {discount_rate*100:.0f}% 할인 프로모션을 발송했습니다. (시뮬레이션)"
                }

            else:
                output = {"success": False, "error": f"알 수 없는 툴입니다: {func_name}"}

        except requests.HTTPError:
            try:
                output = {
                    "success": False,
                    "error": f"API 오류: {res.status_code}",
                    "detail": res.json()
                }
            except Exception:
                output = {
                    "success": False,
                    "error": f"API 오류: {res.status_code}",
                    "detail": res.text
                }

        except Exception as e:
            output = {
                "success": False,
                "error": f"API 통신 실패: {str(e)}. 서버가 켜져 있는지 확인하세요."
            }

        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": json.dumps(output, ensure_ascii=False)
        })

    return tool_outputs

# ==========================================
# 5. 메인 실행 루프
# ==========================================
def run_conversation():
    assistant = setup_agent()
    thread = client.beta.threads.create()
    
    print("========================================")
    print("마케팅 에이전트가 준비되었습니다.")
    print("(종료하려면 'exit' 입력)")
    print("========================================\n")

    while True:
        user_input = input("관리자: ")
        if user_input.lower() == 'exit': break

        client.beta.threads.messages.create(
            thread_id=thread.id, 
            role="user", 
            content=user_input
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id, 
            assistant_id=assistant.id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            
            if run_status.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                print(f"\n에이전트:\n{messages.data[0].content[0].text.value}\n")
                print("-" * 50)
                break
            
            elif run_status.status == "requires_action":
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = execute_ml_logic(tool_calls)
                
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, 
                    run_id=run.id, 
                    tool_outputs=tool_outputs
                )
            
            elif run_status.status in ["failed", "expired", "cancelled"]:
                print(f"\n[오류] 에이전트 실행 실패: {run_status.status}\n")
                break
            
            time.sleep(1)

if __name__ == "__main__":
    run_conversation()
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# csv 업로드
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose="assistants"
)

# 벡터 스토어 생성
vector_store = client.vector_stores.create(
    name="csv.store"
)

# 연결
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id
)
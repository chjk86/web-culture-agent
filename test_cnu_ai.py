import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai",
    api_key=os.environ["TIMELYGPT_API_KEY"],
)
response = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=[{"role": "user", "content": "한 단어로만 답하세요: 대한민국의 수도는?"}],
)
print(response.choices[0].message.content)
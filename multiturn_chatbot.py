import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import YoutubeLoader
from youtube_search import YoutubeSearch

load_dotenv()


@tool
def get_word_count(text: str) -> str:
    """텍스트의 단어 수와 글자 수를 센다."""
    return f"단어 수: {len(text.split())}, 글자 수: {len(text)}"


@tool
def get_current_time_with_timezone(timezone: str) -> str:
    """IANA 시간대 이름(예: 'Asia/Seoul', 'America/New_York')을 받아
    해당 지역의 현재 시간을 알려준다."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"시간대 오류: {e} (예: 'Asia/Seoul', 'America/New_York' 형식으로 입력)"


@tool
def get_web_search(query: str, search_period: str = "m") -> str:
    """웹에서 최신 정보를 검색한다.

    Args:
        query: 검색어
        search_period: 검색 기간 ('d': 하루, 'w': 일주일, 'm': 한 달, 'y': 일 년)
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, results_separator=";\n")
    return search.invoke(query)


@tool
def get_youtube_search(query: str) -> str:
    """유튜브에서 영상을 검색하고, 각 영상의 자막 내용까지 가져온다.
    영상 추천·비교·내용 요약 요청에 사용한다.

    Args:
        query: 검색어
    """
    videos = YoutubeSearch(query, max_results=3).to_dict()
    videos = [v for v in videos if len(v["duration"].split(":")) < 3]

    results = []
    for v in videos:
        video_url = "https://youtube.com" + v["url_suffix"].split("&")[0]
        try:
            docs = YoutubeLoader.from_youtube_url(video_url, language=["ko", "en"]).load()
            content = " ".join(d.page_content for d in docs)[:2000]
        except Exception as error:
            content = f"(자막을 가져오지 못했습니다: {error})"
        results.append(
            f"제목: {v['title']}\n채널: {v['channel']} | 길이: {v['duration']}\n"
            f"URL: {video_url}\n자막(일부): {content}"
        )
    return "\n\n---\n\n".join(results) if results else "검색 결과가 없습니다."


tools = [
    get_word_count,
    get_current_time_with_timezone,
    get_web_search,
    get_youtube_search,
]
tool_map = {t.name: t for t in tools}

MODEL = "google_genai:gemini-flash-latest"
model = init_chat_model(MODEL, temperature=0)
model_with_tools = model.bind_tools(tools)


def get_ai_response(messages: list) -> list:
    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)

    max_rounds = 5
    rounds = 0
    while ai_message.tool_calls and rounds < max_rounds:
        for call in ai_message.tool_calls:
            tool_fn = tool_map[call["name"]]
            result = tool_fn.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        ai_message = model_with_tools.invoke(messages)
        messages.append(ai_message)
        rounds += 1

    return messages


def main():
    print("=== 도구 4개를 가진 멀티턴 챗봇 (종료하려면 'exit' 입력) ===")
    messages = [
        SystemMessage(content=(
            "너는 사용자의 질문에 가장 적합한 도구를 골라 호출해 답하는 AI 조교다. "
            "도구 결과에 없는 내용은 추측하지 마라."
        ))
    ]

    while True:
        user_input = input("\n나: ").strip()
        if user_input.lower() in ("exit", "quit", "종료"):
            print("챗봇을 종료합니다.")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        messages = get_ai_response(messages)

        print("에이전트:", messages[-1].content)


if __name__ == "__main__":
    main()

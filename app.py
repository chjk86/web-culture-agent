import os, re, requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)

# Streamlit Cloud의 Secrets를 os.environ에도 반영 (로컬 .env와 동일하게 동작하도록)
for key, value in st.secrets.items():
    os.environ[key] = str(value)

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient
from youtube_search import YoutubeSearch


def parse_views(view_str):
    num = re.search(r"[\d,\.]+", view_str)
    if not num:
        return 0
    value = float(num.group().replace(",", ""))
    if "만" in view_str:
        value *= 10000
    elif "억" in view_str:
        value *= 100000000
    return int(value)


def linkify(text: str) -> str:
    """텍스트 안의 bare URL을 클릭 가능한 마크다운 링크로 감싼다."""
    return re.sub(r'(https?://[^\s|]+)', r'[\1](\1)', text)


@tool
def search_reception_youtube(query: str, max_results: int = 15) -> str:
    """웹소설·웹툰·드라마·영화·뮤지컬이 유튜브에서 어떻게 대중적으로 재생산·소비되는지
    조회수 순으로 검색한다. 결과에는 video_id와 전체 URL이 포함된다."""
    results = YoutubeSearch(query, max_results=max_results).to_dict()
    videos = sorted(results, key=lambda r: parse_views(r["views"]), reverse=True)
    lines = []
    for v in videos:
        video_id = v["url_suffix"].split("v=")[-1].split("&")[0]
        url = f"https://youtube.com/watch?v={video_id}"
        lines.append(f"[{v['views']}] {v['channel']} | {v['title']} | video_id={video_id} | url={url}")
    return "\n".join(lines) if lines else "검색 결과가 없습니다."


@tool
def get_youtube_comments(video_id: str, max_results: int = 20) -> str:
    """유튜브 영상 ID로 상위 댓글을 가져온다."""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet", "videoId": video_id, "maxResults": max_results,
        "order": "relevance", "key": os.environ["YOUTUBE_API_KEY"],
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return f"댓글 조회 실패: {resp.status_code}"
    data = resp.json()
    comments = [
        f"[좋아요{item['snippet']['topLevelComment']['snippet']['likeCount']}] "
        f"{item['snippet']['topLevelComment']['snippet']['textOriginal']}"
        for item in data.get("items", [])
    ]
    return "\n---\n".join(comments) if comments else "댓글이 없습니다."


@tool
def search_criticism_web(topic: str) -> str:
    """작품에 대한 평론·비평 기사를 웹에서 검색한다."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    queries = [f"{topic} 평론", f"{topic} 비평 분석", f"{topic} 재현 리뷰"]
    exclude_domains = ["namu.wiki", "wikipedia.org", "blog.naver.com", "youtube.com"]
    review_keywords = ["평론", "비평", "리뷰", "분석", "칼럼", "시선", "재현"]
    seen_urls, candidates = set(), []
    for q in queries:
        res = tavily_client.search(
            query=q, max_results=5, search_depth="basic", topic="general",
            include_answer=False, include_raw_content=False,
        )
        for r in res["results"]:
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])
            if any(d in r["url"] for d in exclude_domains):
                continue
            if any(k in r["title"] for k in review_keywords) or r["score"] > 0.5:
                candidates.append(r)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    lines = [f"[{r['score']:.2f}] {r['title']} | {r['url']}" for r in candidates[:8]]
    return "\n".join(lines) if lines else "평론 성격의 결과를 찾지 못했습니다."


@tool
def extract_article(urls: list) -> str:
    """평론 URL들의 본문을 정제해서 추출한다."""
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    extracted = tavily_client.extract(urls=urls)
    parts = [
        f"URL: {item['url']}\n본문: {item['raw_content'][:1500]}"
        for item in extracted["results"]
    ]
    return "\n\n---\n\n".join(parts) if parts else "본문 추출에 실패했습니다."


@st.cache_resource
def build_app():
    tools = [search_reception_youtube, get_youtube_comments, search_criticism_web, extract_article]
    model = init_chat_model("google_genai:gemini-flash-latest", temperature=0.2)
    model_with_tools = model.bind_tools(tools)

    SYSTEM_PROMPT = SystemMessage(content=(
        "너는 웹소설·웹툰·드라마·영화·뮤지컬 등 대중서사가 어떻게 대중적으로 재생산·소비되고 "
        "동시에 어떻게 비평적으로 논의되는지를 함께 조사하는 연구 보조원이다. "
        "필요하면 여러 도구를 순차적으로 사용해 충분한 근거를 모은 뒤 종합해서 답하라."
    ))

    def agent_node(state):
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


st.title("웹문화연구 보조 Agent")
st.caption("웹소설·웹툰·드라마·영화·뮤지컬의 대중 반응(유튜브)과 평론을 함께 조사합니다.")

app = build_app()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("예: 옥씨부인전 대중 반응과 평론 비교해줘"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("검색하고 분석하는 중..."):
            history_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.chat_history[:-1]
            ]
            result = app.invoke(
                {"messages": history_messages + [HumanMessage(content=prompt)]},
                config={"recursion_limit": 15},
            )

            with st.expander("🔍 원문 보기 (검색 결과·댓글 원본)"):
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for call in msg.tool_calls:
                            st.markdown(f"**호출: `{call['name']}`** (인자: `{call['args']}`)")
                    if isinstance(msg, ToolMessage):
                        st.markdown(linkify(msg.content))
                        st.markdown("---")

            content = result["messages"][-1].content
            if isinstance(content, list):
                text = "\n".join(
                    part["text"] for part in content if isinstance(part, dict) and "text" in part
                )
            else:
                text = content
            st.markdown(text)

    st.session_state.chat_history.append({"role": "assistant", "content": text})
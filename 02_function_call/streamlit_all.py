from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import streamlit as st

from langchain_core.tools import tool
from datetime import datetime
import pytz
import yfinance as yf

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from youtube_search import YoutubeSearch
from langchain_community.document_loaders import YoutubeLoader
from typing import List

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"
    
@tool
def get_web_search(query: str, search_period: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
        search_period (str): 검색 기간 (e.g., "w" for past week, "m" for past month, "y" for past year)

    Returns:
        str: 검색 결과
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)

    print('-------- WEB SEARCH --------')
    print(query)
    print(search_period)

    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        # source="news",
        results_separator=';\n'
    )

    docs = search.invoke(query)
    return docs

@tool
def get_youtube_search(query: str) -> List:
    """
    유튜브 검색을 한 뒤, 영상들의 내용을 반환하는 함수.

    Args:
        query (str): 검색어

    Returns:
        List: 검색 결과
    """
    print('-------- YOUTUBE SEARCH --------')
    print(query)

    videos = YoutubeSearch(query, max_results=5).to_dict()

    # 1시간 이상의 영상은 스킵 (59:59가 최대 길이)
    videos = [video for video in videos if len(f"{video['duration']}") <= 4]

    for video in videos:
        video_url = 'http://youtube.com' + video['url_suffix'] # 영상 URL

        loader = YoutubeLoader.from_youtube_url(
            video_url, 
            language=['ko', 'en'] # 자막 언어
        )
        
        video['video_url'] = video_url
        video['content'] = loader.load()

    return videos

@tool
def get_yf_stock_info(ticker: str):
    """
    주식 종목의 정보를 가져옵니다.

    Args:
        ticker (str): 주식 종목의 티커

    Returns:
        str: 주식 정보
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    print(info)
    return str(info)

@tool
def get_yf_stock_history(ticker: str, period: str):
    """
    주식의 히스토리를 가져오는 함수.
    최근 기간의 주식 가격 정보를 가져옵니다.

    Args:
        ticker (str): 주식 종목의 티커
        period (str): 기간 (e.g., "1d" for 1 day, "1mo" for 1 month, "1y" for 1 year)

    Returns:
        str: 주식 히스토리
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    history_md = history.to_markdown() # 데이터프레임을 마크다운 형식으로 변환
    print(history_md)
    return history_md

@tool
def get_yf_stock_recommendations(ticker: str):
    """
    주식 추천 정보를 가져오는 함수.
    글로벌 금융 기관의 추천 정보를 가져옵니다.

    Args:
        ticker (str): 주식 종목의 티커

    Returns:
        str: 주식 추천 정보
    """
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    recommendations_md = recommendations.to_markdown() # 데이터프레임을 마크다운 형식으로 변환
    print(recommendations_md)
    return recommendations_md

# 도구 바인딩
tools = [
    get_current_time, 
    get_web_search, 
    get_youtube_search,
    get_yf_stock_info,
    get_yf_stock_history,
    get_yf_stock_recommendations,
]
tool_dict = {
    "get_current_time": get_current_time,
    "get_web_search": get_web_search,
    "get_youtube_search": get_youtube_search,
    "get_yf_stock_info": get_yf_stock_info,
    "get_yf_stock_history": get_yf_stock_history,
    "get_yf_stock_recommendations": get_yf_stock_recommendations,
}

llm_with_tools = model.bind_tools(tools)

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)
    
    gathered = None
    for chunk in response:
        yield chunk
        
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
 
    
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg, type(tool_msg))
            st.session_state.messages.append(tool_msg)
            
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk
        

# Streamlit 앱
st.title("💬 GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다. ")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    with st.spinner("AI가 답변을 준비 중입니다..."):
        response = get_ai_response(st.session_state["messages"])
    
    result = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(result)) # AI 메시지 저장 


"""LangGraph Adım 1: create_agent'ın arka planda kurduğu döngüyü elle kurmak.

    START → model ──(tool çağrısı var mı?)──→ tools → model → ...
                  └──────────(yok)──────────→ END
"""
import json
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from shared import job_source
from shared.config import MODEL
from shared.prompts import SYSTEM_PROMPT


# --- Context ve State ------------------------------------------------------
@dataclass
class Context:
    resume: str


class State(TypedDict):
    # add_messages bir "reducer": node {"messages": [yeni]} döndürünce liste
    # değiştirilmez, yeni mesajlar sona eklenir. create_agent'taki AgentState bunu
    # hazır veriyordu; burada kendimiz tanımlıyoruz.
    messages: Annotated[list[AnyMessage], add_messages]


# --- Tools -------------------------------------------------------------------
@tool
def get_my_profile() -> str:
    """Kullanıcının CV'sini ve iş tercihlerini döndürür. Kullanıcıya uygun ilan aramadan önce çağır."""
    # Gövde kullanılmıyor: tool_node bu çağrıyı context'teki CV ile cevaplıyor.
    raise NotImplementedError


@tool
def search_jobs(keywords: str) -> str:
    """İş ilanlarında arama yapar.

    Args:
        keywords: Rol veya yetenek anahtar kelimeleri, boşlukla ayrılmış. Kısaltma yerine
            açık yaz ve eş anlamlıları ekle (örn. "machine learning ml llm nlp").
    """
    results = job_source.search_jobs(keywords)
    return json.dumps(results, ensure_ascii=False) if results else "Eşleşen ilan bulunamadı."


TOOLS = [get_my_profile, search_jobs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

PROMPT = SYSTEM_PROMPT + """

Kişisel bir istekte önce get_my_profile ile profili oku, sonra search_jobs'u tek seferde
tüm hedef rollerle çağır. İlanları id'leriyle listele."""

# create_agent'a model adını string verip tools=[...] diyorduk; o bizim yerimize
# init_chat_model + bind_tools yapıyordu.
model = init_chat_model(MODEL).bind_tools(TOOLS)


# --- Node'lar ----------------------------------------------------------------
def call_model(state: State) -> dict:
    """Model node: system prompt + geçmiş → bir AI mesajı (metin ya da tool çağrısı)."""
    response = model.invoke([SystemMessage(PROMPT), *state["messages"]])
    return {"messages": [response]}


def run_tools(state: State, runtime: Runtime[Context]) -> dict:
    """Tool node: son AI mesajındaki her tool çağrısını çalıştırıp ToolMessage döner."""
    results = []
    for call in state["messages"][-1].tool_calls:
        if call["name"] == "get_my_profile":
            # create_agent'ta ToolRuntime bunu otomatik yapıyordu: context'i tool'a enjekte etmek.
            output = runtime.context.resume
        else:
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        results.append(ToolMessage(output, tool_call_id=call["id"], name=call["name"]))
    return {"messages": results}


def should_continue(state: State) -> str:
    """Conditional edge: model tool çağırdıysa tools'a, çağırmadıysa bitişe."""
    return "tools" if state["messages"][-1].tool_calls else END


# --- Graph -------------------------------------------------------------------
def build_graph(checkpointer=None):
    builder = StateGraph(State, context_schema=Context)

    # create_agent'taki ModelRetryMiddleware'ın karşılığı: node seviyesinde retry policy
    builder.add_node("model", call_model,
                     retry_policy=RetryPolicy(max_attempts=5, initial_interval=2.0))
    builder.add_node("tools", run_tools)

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", should_continue, ["tools", END])
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=checkpointer)
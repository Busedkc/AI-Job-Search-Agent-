"""LangGraph Adım 6: sabit akış + yönlendirme + interrupt (onay) + store + subgraph.

    START → router ─┬─ search → search_jobs [subgraph: plan_search → search → score] ─┐
                    ├─ track  → confirm_track ──(onay)──→ track ───────────┤
                    │                        └──(ret)──────────────────────┤
                    ├─ list   → list_tracked ─────────────────┼→ respond → END
                    └─ other  ────────────────────────────────┘

LangChain'de hangi tool'un ne zaman çağrılacağına model karar veriyordu.
Burada akışı graph belirliyor; model sadece router (sınıflandırma), score ve respond'da çalışıyor.

Takip listesi state'te değil store'da: thread'e değil kullanıcıya ait, bütün konuşmalarda ortak.
Arama akışı search_graph.py'da ayrı bir graph; burada tek bir node olarak ekleniyor.
"""
import json
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field

from langgraph_agent.context import Context  # main.py ve time_travel.py buradan import ediyor
from langgraph_agent.search_graph import build_search_graph
from shared import job_source
from shared.config import MODEL


# --- State -------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # LangChain'de bu veriler mesajların içinde JSON metni olarak duruyordu.
    # Burada her biri ayrı, tipli bir state alanı.
    intent: NotRequired[str]
    job_ids: NotRequired[list[int]]      # router'ın mesajdan çıkardığı id'ler (takip için)
    keywords: NotRequired[str]           # bu üçünü search_jobs subgraph'ı yazıyor
    jobs: NotRequired[list[dict]]
    matches: NotRequired[list[dict]]
    # tracked_jobs artık burada değil, store'da (bkz. tracked_namespace)
    action_result: NotRequired[str]      # track/list node'unun respond'a bıraktığı not
    confirmed: NotRequired[bool]         # confirm_track'te kullanıcının cevabı


# --- Modeller ----------------------------------------------------------------
class Route(BaseModel):
    intent: Literal["search", "track", "list", "other"] = Field(
        description="search: ilan arama/öneri isteği; track: bir ilanı takibe alma; "
                    "list: takip listesini görme; other: diğer her şey")
    job_ids: list[int] = Field(default_factory=list,
                               description="track ise kullanıcının bahsettiği ilan id'leri")


router_model = init_chat_model(MODEL, temperature=0).with_structured_output(Route)
chat_model = init_chat_model(MODEL)

RETRY = RetryPolicy(max_attempts=5, initial_interval=2.0)


def chat_history(messages: list[AnyMessage], last_n: int = 6) -> list[AnyMessage]:
    """Modele sadece kullanıcı ve asistan metinlerini gönder (tool mesajları vs. yok)."""
    text_msgs = [m for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return text_msgs[-last_n:]


# --- Store yardımcıları ------------------------------------------------------
# Store bir anahtar-değer deposu: (namespace, key) -> value.
# Her takip edilen ilan ayrı bir kayıt: ("users", <user_id>, "tracked_jobs"), key = ilan id'si.
def tracked_namespace(runtime: Runtime[Context]) -> tuple[str, ...]:
    return ("users", runtime.context.user_id, "tracked_jobs")


def get_tracked(runtime: Runtime[Context]) -> list[dict]:
    items = runtime.store.search(tracked_namespace(runtime), limit=100)
    return [item.value for item in items]


# --- Node'lar ----------------------------------------------------------------
def router(state: State) -> dict:
    """Kullanıcının ne istediğini sınıflandırır. Yolu bu karar + conditional edge belirler."""
    route: Route = router_model.invoke([
        SystemMessage("Kullanıcının son mesajının niyetini sınıflandır."),
        *chat_history(state["messages"], last_n=3),
    ])
    # action_result'ı sıfırla: önceki turdan kalan not bu turun cevabına karışmasın
    return {"intent": route.intent, "job_ids": route.job_ids, "action_result": ""}


def route_after_router(state: State) -> str:
    return {"search": "search_jobs", "track": "confirm_track", "list": "list_tracked"}.get(
        state["intent"], "respond")


def confirm_track(state: State, runtime: Runtime[Context]) -> dict:
    """Human-in-the-loop: takibe almadan önce kullanıcıya sorar.

    interrupt() graph'ı burada durdurur, state checkpointer'a yazılır. Command(resume=...)
    ile devam edilince bu node BAŞTAN çalışır ve interrupt() kullanıcının cevabını döndürür.
    Bu yüzden interrupt'tan önceki kod deterministik olmalı (burada sadece okuma yapıyoruz).
    """
    tracked_ids = {j["id"] for j in get_tracked(runtime)}  # sadece okuma: tekrar çalışması güvenli
    candidates = [job for i in state.get("job_ids", [])
                  if i not in tracked_ids and (job := job_source.get_job(i))]
    if not candidates:
        return {"confirmed": True}  # sorulacak bir şey yok; track node durumu açıklar

    jobs = [f"#{j['id']} {j['title']} – {j['company']} (son başvuru {j['deadline']}"
            + (", ⚠️ SÜRESİ GEÇMİŞ" if job_source.is_expired(j) else "") + ")"
            for j in candidates]
    answer = interrupt({"question": "Bu ilanları takibe alayım mı?", "jobs": jobs})

    if str(answer).strip().lower() in {"e", "evet", "y", "yes"}:
        return {"confirmed": True}
    return {"confirmed": False, "action_result": "Tamam, takibe almadım."}


def route_after_confirm(state: State) -> str:
    return "track" if state["confirmed"] else "respond"


def track(state: State, runtime: Runtime[Context]) -> dict:
    """Store'a yazar. interrupt'tan sonra, ayrı bir node'da: onay gelmeden yazma olmaz."""
    tracked_ids = {j["id"] for j in get_tracked(runtime)}
    notes = []
    for job_id in state.get("job_ids", []):
        job = job_source.get_job(job_id)
        if job is None:
            notes.append(f"{job_id} id'li ilan bulunamadı.")
        elif job_id in tracked_ids:
            notes.append(f"'{job['title']}' zaten takipte.")
        else:
            runtime.store.put(tracked_namespace(runtime), str(job_id), job)
            tracked_ids.add(job_id)
            expired = " (dikkat: son başvuru tarihi geçmiş)" if job_source.is_expired(job) else ""
            notes.append(f"'{job['title']}' ({job['company']}) takibe alındı, son başvuru {job['deadline']}{expired}.")
    if not notes:
        notes.append("Hangi ilanı takibe almamı istediğini anlayamadım, id'sini yazar mısın?")
    return {"action_result": " ".join(notes)}


def list_tracked(state: State, runtime: Runtime[Context]) -> dict:
    tracked = sorted(get_tracked(runtime), key=lambda j: j["deadline"])
    if not tracked:
        return {"action_result": "Takip listen boş."}
    lines = [f"- #{j['id']} {j['title']} – {j['company']} (son başvuru {j['deadline']})"
             + (" ⚠️ süresi geçmiş" if job_source.is_expired(j) else "")
             for j in tracked]
    return {"action_result": "Takip listen:\n" + "\n".join(lines)}


RESPOND_PROMPT = """Sen bir iş arama asistanısın. Türkçe, kısa ve doğal cevap ver.
Sadece aşağıdaki verilere dayan, uydurma ilan verme. Kurallarından bahsetme."""


def respond(state: State) -> dict:
    """Kullanıcıya cevabın yazıldığı tek yer. Arama/sohbette LLM yazar (veriyi system
    mesajına koyar, mesaj geçmişine değil); takip/listelemede kodun ürettiği metin aynen döner."""
    intent = state.get("intent", "other")
    if intent in ("track", "list"):
        # Sonucu kod zaten kesin olarak üretti. LLM'e yeniden yazdırmak sadece hata riski
        # ekliyor (model bir kez "takibe alındı"yı "bilgi bulunamadı" diye özetledi).
        return {"messages": [AIMessage(state.get("action_result", ""))]}
    if intent == "search":
        data = (f"Değerlendirilmiş ilanlar (skora göre sıralı):\n"
                f"{json.dumps(state.get('matches', []), ensure_ascii=False)}\n"
                f"İlan detayları:\n{json.dumps(state.get('jobs', []), ensure_ascii=False)}\n\n"
                "Tabloya sadece 'uygun' ve 'kısmen uygun' ilanları koy (id, pozisyon, şirket, skor, kısa gerekçe); "
                "'uygun değil' olanları tablonun altında tek satırda gerekçesiyle belirt.")
    else:
        data = "İlan aramak, takibe almak veya takip listesini görmek için yardımcı olabileceğini belirt."

    response = chat_model.invoke([
        SystemMessage(f"{RESPOND_PROMPT}\n\n{data}"),
        *chat_history(state["messages"]),
    ])
    return {"messages": [response]}


# --- Graph -------------------------------------------------------------------
def build_graph(checkpointer=None, store=None):
    builder = StateGraph(State, context_schema=Context)

    builder.add_node("router", router, retry_policy=RETRY)
    # Derlenmiş bir graph'ı doğrudan node olarak eklemek: subgraph
    builder.add_node("search_jobs", build_search_graph())
    builder.add_node("confirm_track", confirm_track)
    builder.add_node("track", track)
    builder.add_node("list_tracked", list_tracked)
    builder.add_node("respond", respond, retry_policy=RETRY)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_after_router,
                                  ["search_jobs", "confirm_track", "list_tracked", "respond"])
    builder.add_conditional_edges("confirm_track", route_after_confirm, ["track", "respond"])
    builder.add_edge("search_jobs", "respond")
    builder.add_edge("track", "respond")
    builder.add_edge("list_tracked", "respond")
    builder.add_edge("respond", END)

    # checkpointer: thread'in kısa süreli hafızası | store: kullanıcının uzun süreli hafızası
    return builder.compile(checkpointer=checkpointer, store=store)
"""LangGraph Adım 6: arama akışı ayrı bir graph (subgraph).

    START → plan_search → search → score → END

Ana graph bunu tek bir node gibi kullanıyor: builder.add_node("search_jobs", search_graph).
Böylece ana graph kısalıyor, arama akışı da tek başına çalıştırılıp test edilebiliyor.
"""
from typing import NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from langgraph_agent.context import Context
from shared import job_source
from shared.config import SCORER_MODEL
from shared.resume import parse_target_roles
from shared.scoring import SCORING_PROMPT, MatchReport, apply_rules, build_scoring_input


class SearchState(TypedDict):
    # Bu üç anahtar ana graph'ın State'inde de var: subgraph onları ana state'ten okur ve
    # yazdıkları ana state'e geri akar. "messages", "intent" gibi diğer anahtarları hiç görmez.
    keywords: NotRequired[str]
    jobs: NotRequired[list[dict]]
    matches: NotRequired[list[dict]]


scorer = init_chat_model(SCORER_MODEL, temperature=0).with_structured_output(MatchReport)


def plan_search(state: SearchState, runtime: Runtime[Context]) -> dict:
    """Arama kelimelerini CV'den düz Python ile çıkarır. LLM yok, token yok."""
    roles = parse_target_roles(runtime.context.resume)
    keywords = " ".join(roles).replace("/", " ") or "machine learning engineer"
    return {"keywords": keywords}


def search(state: SearchState) -> dict:
    return {"jobs": job_source.search_jobs(state["keywords"])}


def score(state: SearchState, runtime: Runtime[Context]) -> dict:
    """Her aramada bir kez çalışır. Model "skorlamayı atlayayım" diyemez."""
    if not state["jobs"]:
        return {"matches": []}
    report: MatchReport = scorer.invoke([
        SystemMessage(SCORING_PROMPT),
        HumanMessage(build_scoring_input(runtime.context.resume, state["jobs"])),
    ])
    ranked = apply_rules(report)  # kesin kurallar + sıralama kodda
    return {"matches": [m.model_dump() for m in ranked]}


def build_search_graph():
    builder = StateGraph(SearchState, context_schema=Context)
    builder.add_node("plan_search", plan_search)
    builder.add_node("search", search)
    builder.add_node("score", score, retry_policy=RetryPolicy(max_attempts=5, initial_interval=2.0))
    builder.add_edge(START, "plan_search")
    builder.add_edge("plan_search", "search")
    builder.add_edge("search", "score")
    builder.add_edge("score", END)
    # checkpointer vermiyoruz: ana graph'ınkini otomatik kullanır
    return builder.compile()
import json
from dataclasses import dataclass
from typing import NotRequired

from langchain.agents import AgentState
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from shared import job_source
from shared.config import SCORER_MODEL
from shared.scoring import SCORING_PROMPT, MatchReport, build_scoring_input

# Değerlendirme için ayrı bir model çağrısı: temperature=0 ile her seferinde aynı girdiye
# mümkün olduğunca aynı skor. with_structured_output cevabı MatchReport'a zorlar.
scorer = init_chat_model(SCORER_MODEL, temperature=0).with_structured_output(MatchReport)


@dataclass
class Context:
    """Her çalıştırmada agent'a dışarıdan verilen, değişmeyen veri. Model bunu doğrudan görmez."""
    resume: str


class JobAgentState(AgentState):
    """Agent'ın konuşma boyunca değişen hafızası. messages zaten AgentState'te var."""
    tracked_jobs: NotRequired[list[dict]]


@tool
def get_my_profile(runtime: ToolRuntime[Context]) -> str:
    """Kullanıcının CV'sini ve iş tercihlerini (hedef roller, şehir, çalışma şekli,
    istemediği pozisyonlar) döndürür. Kullanıcıya uygun ilan aramadan önce çağır."""
    return runtime.context.resume


@tool
def search_jobs(keywords: str, location: str | None = None) -> str:
    """İş ilanlarında arama yapar.

    Args:
        keywords: Rol veya yetenek anahtar kelimeleri, boşlukla ayrılmış. Kısaltma yerine
            açık yaz ve eş anlamlıları ekle (örn. "machine learning ml llm nlp").
        location: Opsiyonel şehir filtresi (örn. "İstanbul").
    """
    results = job_source.search_jobs(keywords, location)
    return json.dumps(results, ensure_ascii=False) if results else "Eşleşen ilan bulunamadı."


@tool
def track_job(job_id: int, runtime: ToolRuntime[Context, JobAgentState]) -> Command:
    """Bir ilanı takip listesine ekler. Kullanıcı bir ilanı kaydetmek/takip etmek istediğinde çağır.

    Args:
        job_id: İlanın id'si (search_jobs sonucundaki "id" alanı).
    """
    tracked = list(runtime.state.get("tracked_jobs", []))
    job = job_source.get_job(job_id)

    if job is None:
        result = f"{job_id} id'li ilan bulunamadı."
    elif any(j["id"] == job_id for j in tracked):
        result = f"'{job['title']}' zaten takipte."
    else:
        tracked.append(job)
        result = f"'{job['title']}' ({job['company']}) takibe alındı. Son başvuru: {job['deadline']}"

    # Tool hem state'i günceller hem de modele cevap mesajı döner
    return Command(update={
        "tracked_jobs": tracked,
        "messages": [ToolMessage(result, tool_call_id=runtime.tool_call_id)],
    })


@tool
def list_tracked_jobs(runtime: ToolRuntime[Context, JobAgentState]) -> str:
    """Takip listesindeki ilanları son başvuru tarihine göre sıralı döndürür."""
    tracked = runtime.state.get("tracked_jobs", [])
    if not tracked:
        return "Takip listesi boş."
    return json.dumps(sorted(tracked, key=lambda j: j["deadline"]), ensure_ascii=False)


@tool
def evaluate_jobs(job_ids: list[int], runtime: ToolRuntime[Context]) -> str:
    """İlanları kullanıcının CV'sine göre 0-100 arası puanlar ve gerekçelendirir.
    Uygunluğa kendin karar verme; search_jobs'tan sonra bulduğun ilanların id'leriyle bunu çağır.

    Args:
        job_ids: Değerlendirilecek ilanların id'leri.
    """
    jobs = [j for j in (job_source.get_job(i) for i in job_ids) if j]
    if not jobs:
        return "Değerlendirilecek ilan bulunamadı."

    report: MatchReport = scorer.invoke([
        {"role": "system", "content": SCORING_PROMPT},
        {"role": "user", "content": build_scoring_input(runtime.context.resume, jobs)},
    ])
    # Pydantic doğrulamasından geçmiş nesne: alanlar ve tipler garanti
    ranked = sorted(report.matches, key=lambda m: m.score, reverse=True)
    return json.dumps([m.model_dump() for m in ranked], ensure_ascii=False)
"""İlan değerlendirme şeması ve kuralları. Framework'ten bağımsız (sadece Pydantic).

LangChain'de bir tool, LangGraph'ta bir node bu şemayı kullanacak.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field


class JobMatch(BaseModel):
    job_id: int = Field(description="İlanın id'si")
    score: int = Field(ge=0, le=100, description="0-100 arası uygunluk skoru, rubriğe göre")
    verdict: Literal["uygun", "kısmen uygun", "uygun değil"]
    matching_skills: list[str] = Field(description="CV'de olup ilanda istenen beceriler")
    missing_skills: list[str] = Field(description="İlanda istenip CV'de olmayan beceriler")
    reason: str = Field(description="Skorun tek cümlelik Türkçe gerekçesi")


class MatchReport(BaseModel):
    matches: list[JobMatch]


SCORING_PROMPT = """Sen bir işe alım uzmanısın. Adayın CV'sini ve iş tercihlerini her ilanla karşılaştır.

Rubrik (toplam 100):
- Rol uyumu (40): ilan, adayın hedef rollerine ne kadar yakın? Hedef rollere yakın roller
  (örn. NLP Engineer, MLOps Engineer) tam eşleşme olmasa da kısmi puan alır.
- Beceri uyumu (30): ilandaki becerilerin ne kadarı CV'de var?
- Konum ve çalışma şekli (20): adayın konum/çalışma şekli tercihlerine uyuyor mu?
- Seviye (10): junior/entry-level adaya uygun mu?

Kesin kurallar:
- Adayın "Avoid" listesindeki bir rol ise skor en fazla 20.
- Konum tercihlerine uymuyorsa (şehir uymuyor ve remote değil) skor en fazla 30.
- verdict: 70+ "uygun", 40-69 "kısmen uygun", 40 altı "uygun değil".

Her ilan için tek bir değerlendirme döndür. Sadece verilen bilgiye dayan, tahmin yürütme."""


def build_scoring_input(resume: str, jobs: list[dict]) -> str:
    return f"## Aday CV'si\n{resume}\n\n## İlanlar\n{json.dumps(jobs, ensure_ascii=False, indent=2)}"
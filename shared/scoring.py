"""İlan değerlendirme şeması ve kuralları. Framework'ten bağımsız (sadece Pydantic).

LangChain'de bir tool, LangGraph'ta bir node bu şemayı kullanacak.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field


class JobMatch(BaseModel):
    job_id: int = Field(description="İlanın id'si")
    # Önce tespitler (LLM evet/hayır der), sonra skor. Kesin kuralları kod uygular: apply_rules()
    location_ok: bool = Field(description="İlan adayın konum/çalışma şekli tercihlerine uyuyor mu? "
                                          "(şehir tercih listesinde VEYA ilan remote ise true)")
    avoided_role: bool = Field(description="İlan adayın 'Avoid' listesindeki bir rol mü?")
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

location_ok ve avoided_role alanlarını dikkatle ve dürüstçe doldur; skor sınırlarını
bunlara göre sistem ayrıca uygulayacak.
verdict: 70+ "uygun", 40-69 "kısmen uygun", 40 altı "uygun değil".

Her ilan için tek bir değerlendirme döndür. Sadece verilen bilgiye dayan, tahmin yürütme."""


def build_scoring_input(resume: str, jobs: list[dict]) -> str:
    return f"## Aday CV'si\n{resume}\n\n## İlanlar\n{json.dumps(jobs, ensure_ascii=False, indent=2)}"


def verdict_for(score: int) -> str:
    return "uygun" if score >= 70 else "kısmen uygun" if score >= 40 else "uygun değil"


def apply_rules(report: MatchReport) -> list[JobMatch]:
    """Kesin kuralları kodla uygular ve skora göre sıralar.

    LLM "konum uymuyor" diye yazıp yine 65 verebiliyor; sınırı prompt'a bırakmak güvenilir değil.
    """
    final = []
    for m in report.matches:
        score = m.score
        if m.avoided_role:
            score = min(score, 20)
        if not m.location_ok:
            score = min(score, 30)
        final.append(m.model_copy(update={"score": score, "verdict": verdict_for(score)}))
    return sorted(final, key=lambda m: m.score, reverse=True)
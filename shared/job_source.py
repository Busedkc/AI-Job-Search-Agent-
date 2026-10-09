"""İlan kaynağı. Framework'ten bağımsız saf Python.
LangChain ve LangGraph versiyonları bu fonksiyonları kendi tool/node'larına sarar.
Mock'tan gerçek API'ye geçince sadece bu dosya değişecek.
"""
import json

from shared.config import DATA_DIR


def load_jobs() -> list[dict]:
    with open(DATA_DIR / "mock_jobs.json", encoding="utf-8") as f:
        return json.load(f)


def search_jobs(keywords: str, location: str | None = None) -> list[dict]:
    words = [w.lower() for w in keywords.split()]
    results = []
    for job in load_jobs():
        haystack = (job["title"] + " " + " ".join(job["skills"])).lower()
        if not any(w in haystack for w in words):
            continue
        if location and location.lower() not in job["location"].lower():
            continue
        results.append(job)
    return results


def get_job(job_id: int) -> dict | None:
    return next((j for j in load_jobs() if j["id"] == job_id), None)
# AI Job Search Agent: LangChain vs LangGraph

Aynı iş arama agent'ı iki kez yapılıyor: önce LangChain (`create_agent`), sonra LangGraph (açık graph).

## Yapı

```
shared/            # framework'ten bağımsız, iki versiyon da kullanır
  config.py        # .env ve model ayarı
  job_source.py    # ilan arama mantığı (şimdilik mock)
  prompts.py       # ortak system prompt
  data/            # mock ilanlar
langchain_agent/   # LangChain versiyonu
langgraph_agent/   # LangGraph versiyonu
```

Kural: `shared/` içinde langchain/langgraph import edilmez. Framework'e özel olan her şey kendi klasöründe.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # key'leri doldur
```

## Çalıştırma (repo kökünden)

```bash
python -m langchain_agent.agent
```

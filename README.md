# AI Job Search Agent: LangChain vs LangGraph

CV'ni okuyup sana uygun iş ilanlarını bulan, puanlayan ve takip listesinde tutan bir agent.
Aynı agent'ı iki kez yazdım: önce **LangChain** (`create_agent`) ile, sonra **LangGraph** ile
açık bir graph olarak. Amaç iki yaklaşımın farkını aynı problem üzerinde görmekti.

> İlanlar şimdilik `shared/data/mock_jobs.json` içindeki örnek verilerden geliyor.
> Gerçek ilan kaynağı bir sonraki adım.

## Ne yapıyor?

- CV'ndeki hedef rollere göre ilan arıyor
- Her ilanı 0-100 arası puanlıyor, gerekçesini ve eksik becerileri yazıyor
- İstediğin ilanları takip listesine alıyor (LangGraph'ta önce onay istiyor)
- Süresi geçmiş ilanları işaretliyor
- Konuşmayı ve takip listesini Postgres'te saklıyor; program kapansa da kaldığı yerden devam ediyor

## Kullanılanlar

| | |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`, ücretsiz katman) |
| Agent | LangChain 1.x, LangGraph 1.x |
| Veritabanı | PostgreSQL 17 (Docker) — checkpointer + store |
| Şema doğrulama | Pydantic |
| Görselleştirme | LangGraph Studio (`langgraph dev`) |

## Klasör yapısı

```
shared/                 framework'ten bağımsız kısım; iki versiyon da bunu kullanıyor
  config.py             .env ve ayarlar
  job_source.py         ilan arama (şimdilik mock)
  resume.py             CV okuma, hedef rolleri çıkarma
  scoring.py            puanlama şeması, rubrik ve kesin kurallar
  prompts.py            ortak system prompt
  data/                 mock ilanlar, örnek CV

langchain_agent/
  tools.py              tool'lar + state/context tanımları
  agent.py              create_agent + streaming + Postgres

langgraph_agent/
  graph.py              ana graph: router, onay, takip, cevap
  search_graph.py       arama akışı (subgraph): plan_search → search → score
  context.py            Context (CV, kullanıcı id)
  main.py               terminal arayüzü
  time_travel.py        geçmiş bir checkpoint'e dönüp oradan farklı devam etme
  studio.py             langgraph dev için giriş noktası
  loop_graph.py         ilk adım: create_agent döngüsünün elle yazılmış hali (karşılaştırma için)
```

`shared/` içinde langchain/langgraph import edilmiyor. Böylece iki versiyon arasındaki fark
sadece orkestrasyonda kalıyor.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # GROQ_API_KEY'i doldur (https://console.groq.com/keys)
docker compose up -d              # Postgres'i başlatır
```

Kendi CV'ni `shared/data/resume.md` olarak koyabilirsin (git'e gitmez).
Yoksa `resume.example.md` kullanılır.

## Çalıştırma

Repo kökünden:

```bash
# LangChain versiyonu
python -m langchain_agent.agent [thread_adı]

# LangGraph versiyonu
python -m langgraph_agent.main [thread_adı]

# LangGraph: geçmiş bir onay noktasına dön
python -m langgraph_agent.time_travel [thread_adı]
```

Örnek sorular: `bana uygun ilanları bul`, `2'yi takibe al`, `takip listemde ne var`.
LangGraph versiyonunda `/state` yazınca o anki state'i görürsün.

### Studio

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev --no-reload
```

Tarayıcıda `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` açılır
(ücretsiz LangSmith hesabı gerekiyor). Studio kendi geçici deposunu kullanıyor, Postgres'teki
verileri görmez.

Chrome'da "Failed to fetch" alırsan: adres çubuğundaki site ayarlarından
**Apps on device** iznini aç. Safari'de `langgraph dev --tunnel` kullan.

## İki versiyonun akışı

**LangChain** — model her adımda hangi tool'u çağıracağına kendisi karar veriyor:

```
model → get_my_profile → model → search_jobs → model → evaluate_jobs → model → cevap
```

**LangGraph** — akışı graph belirliyor, model sadece gereken yerde çalışıyor:

```
router ─┬─ search → [plan_search → search → score] ──────────┐
        ├─ track  → confirm_track ─(onay)→ track ───────────┤
        ├─ list   → list_tracked ───────────────────────────┼→ respond
        └─ other  ──────────────────────────────────────────┘
```

## Karşılaştırma

| | LangChain (`create_agent`) | LangGraph (açık graph) |
|---|---|---|
| Akışı kim belirliyor | Model | Graph (router + edge'ler) |
| Kod miktarı | Az; döngü, tool çalıştırma, state hazır | Fazla; her şeyi kendin tanımlıyorsun |
| Bir aramadaki model çağrısı | ~5 (her tool kararı ayrı çağrı + skorlama) | 3 (router, skorlama, cevap) |
| 4 tur sonra mesaj sayısı | 18 (tool çağrıları ve sonuçları dahil) | 8 (sadece soru-cevap) |
| Tutarlılık | Model bazen tool'u atlıyor, aramayı bölüyor | Arama ve skorlama her seferinde aynı sırayla çalışıyor |
| Esneklik | "Sadece remote olanlar" gibi istekleri kendisi yorumlar | Router'ın bilmediği istekler için kod eklemek gerekir |
| Hata/retry | `ModelRetryMiddleware`, `ToolRetryMiddleware` | Node başına `RetryPolicy` |
| İnsan onayı | `HumanInTheLoopMiddleware` (tool öncesi) | `interrupt()` istenen her yerde, soruyu sen tasarlıyorsun |
| Hafıza | Aynı checkpointer ve store | Aynı checkpointer ve store |

İkisi de altta aynı LangGraph motorunu kullanıyor. `create_agent` aslında hazır kurulmuş
bir graph (`model` ve `tools` node'ları). Streaming'deki `langgraph_node` alanı da bunu gösteriyor.

**Kısaca:** Esnek, açık uçlu bir sohbet için LangChain hızlı ve yeterli. Adımları belli,
her seferinde aynı şekilde çalışması gereken ve onay isteyen bir akış için LangGraph daha uygun.

## Yol boyunca öğrendiklerim

- **Tool açıklaması da prompt'un parçası.** `search_jobs` docstring'ine "kısaltma yerine açık yaz"
  ekleyince model "ML" yerine "machine learning" aramaya başladı ve kaçan ilanı buldu.
- **Kesin kuralları LLM'e bırakma.** Rubrikte "konum uymuyorsa en fazla 30" yazmasına rağmen model
  bir keresinde 65 verdi. Çözüm: LLM sadece tespit yapıyor (`location_ok`, `avoided_role`),
  sınırı `apply_rules()` kodla uyguluyor.
- **Sonucu kod biliyorsa LLM'e yeniden yazdırma.** Takip işlemi başarılıyken cevap node'u
  "ilan bulunamadı" diye özetledi. Takip/listeleme cevapları artık doğrudan koddan geliyor.
- **Her adım LLM gerektirmiyor.** Hedef rolleri CV'den çıkarmak düz Python ile yapılıyor;
  token harcamıyor.
- **Mesaj geçmişi büyüdükçe maliyet artıyor.** LangChain'de CV ve ilan JSON'ları her turda
  modele tekrar gidiyor; Groq'un dakikalık token limitine bu yüzden takıldım.
- **interrupt'tan sonra node baştan çalışıyor.** Bu yüzden onay node'u sadece okuma yapıyor,
  yazma işlemi ayrı `track` node'unda.
- **Time travel için fork gerekiyor.** Eski checkpoint'e doğrudan `Command(resume=...)` gönderince
  eski cevap tekrar kullanıldı. `update_state` ile yeni dal açınca soru yeniden soruldu.
- **Time travel store'u geri sarmıyor.** Konuşma geri alınıyor ama takip listesi (store) aynı kalıyor.
  Geri alınamayan işlemler bu yüzden onaydan geçmeli.

## Bilinen sınırlar

- İlanlar mock veri; gerçek kaynak henüz bağlı değil
- Groq ücretsiz katmanında dakikalık token limiti var; yoğun kullanımda retry devreye giriyor
- Router sadece dört niyeti tanıyor (ara, takip et, listele, diğer)

## Sıradaki adımlar

- Gerçek ilan kaynağı (JSearch, Remotive) ve tekrar eden ilanları ayıklama
- Başvuru durumu takibi (başvurdum / mülakat / ret) ve deadline hatırlatması
- Günlük otomatik tarama ve bildirim
- Streamlit arayüzü
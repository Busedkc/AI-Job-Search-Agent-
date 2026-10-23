import sys

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command

from langgraph_agent.graph import Context, build_graph
from shared.config import DATABASE_URL, USER_ID
from shared.resume import load_resume


def describe_update(node: str, update: dict) -> str | None:
    """Her node bitince ekrana tek satırlık özet."""
    if node == "search_jobs":
        return f"📦 search_jobs (subgraph) bitti → {len(update.get('matches', []))} değerlendirme ana state'e geçti"
    return _describe(node, update)


def _describe(node: str, update: dict) -> str | None:
    """Her node bitince ekrana tek satırlık özet."""
    if node == "router":
        return f"🔀 router: niyet = {update['intent']}" + (f", id'ler = {update['job_ids']}" if update.get("job_ids") else "")
    if node == "plan_search":
        return f"📝 plan_search: '{update['keywords']}'"
    if node == "search":
        return f"🔎 search: {len(update['jobs'])} ilan bulundu"
    if node == "score":
        return "📊 score: " + ", ".join(f"#{m['job_id']}={m['score']}" for m in update["matches"])
    if node == "__interrupt__":
        return "✋ confirm_track: onay bekleniyor"
    if node == "confirm_track":
        return "✅ confirm_track: devam" if update.get("confirmed") else "❌ confirm_track: vazgeçildi"
    if node == "track":
        return "📌 track: store işlendi"
    if node == "list_tracked":
        return "📌 list_tracked: liste hazırlandı"
    return None


def stream_graph(graph, graph_input, context: Context, config: dict) -> None:
    """graph_input: yeni soru için {"messages": [...]}, interrupt'tan devam için Command(resume=...)."""
    streamed = False  # respond LLM ile mi yazdı (token geldi mi)?
    for chunk in graph.stream(
        graph_input,
        config=config,
        context=context,
        stream_mode=["messages", "updates"],
        subgraphs=True,  # subgraph içindeki node'ların olaylarını da getir
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            # router ve score da model çağırıyor ama onların çıktısı JSON; sadece respond'u akıt
            if metadata["langgraph_node"] == "respond" and token.content:
                print(token.content, end="", flush=True)
                streamed = True

        elif chunk["type"] == "updates":
            # ns: olayın hangi graph'tan geldiği. () = ana graph, ("search_jobs:<id>",) = subgraph
            indent = "    ↳ " if chunk["ns"] else ""
            for node, update in chunk["data"].items():
                if node == "respond" and not streamed:
                    print(update["messages"][-1].content)  # LLM'siz cevap: token akmadı, tek seferde bas
                    continue
                line = describe_update(node, update if isinstance(update, dict) else {})
                if line:
                    print(indent + line)
    print()


if __name__ == "__main__":
    thread_id = "lg-" + (sys.argv[1] if len(sys.argv) > 1 else "default")
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(resume=load_resume(), user_id=USER_ID)
    tracked_ns = ("users", USER_ID, "tracked_jobs")

    # İki ayrı Postgres bağlantısı: checkpointer (thread hafızası) ve store (kullanıcı hafızası)
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer, \
         PostgresStore.from_conn_string(DATABASE_URL) as store:
        checkpointer.setup()
        store.setup()  # store tablolarını oluşturur; varsa dokunmaz
        graph = build_graph(checkpointer, store)

        try:
            print(graph.get_graph(xray=True).draw_ascii())  # xray: subgraph içini de çiz
        except ImportError:
            print(graph.get_graph(xray=True).draw_mermaid())  # grandalf yoksa Mermaid metni
        state = graph.get_state(config).values
        print(f"Thread: {thread_id} | kayıtlı mesaj: {len(state.get('messages', []))} | "
              f"kullanıcı: {USER_ID} | takipte (store): {len(store.search(tracked_ns, limit=100))} ilan")
        print("Çıkmak için 'q', state'i görmek için '/state' yaz.\n")

        while True:
            question = input("Sen: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if question == "/state":
                s = graph.get_state(config).values
                print(f"mesaj: {len(s.get('messages', []))} | intent: {s.get('intent')}")
                print(f"matches: {[(m['job_id'], m['score']) for m in s.get('matches', [])]}")
                print(f"store {tracked_ns}: {[item.key for item in store.search(tracked_ns, limit=100)]}\n")
                continue
            print("Agent:")
            stream_graph(graph, {"messages": [{"role": "user", "content": question}]}, context, config)

            # Graph interrupt'ta durduysa: soruyu göster, cevabı al, kaldığı yerden devam et
            while interrupts := graph.get_state(config).interrupts:
                payload = interrupts[0].value
                print(f"\n{payload['question']}")
                for job in payload["jobs"]:
                    print(f"  {job}")
                answer = input("(e/h): ").strip()
                stream_graph(graph, Command(resume=answer), context, config)
            print()
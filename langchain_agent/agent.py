import sys

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_agent.tools import (
    Context,
    JobAgentState,
    get_my_profile,
    list_tracked_jobs,
    search_jobs,
    track_job,
)
from shared.config import DATABASE_URL, MODEL
from shared.prompts import SYSTEM_PROMPT
from shared.resume import load_resume

# Tool kullanım talimatı LangChain'e özel, o yüzden shared prompt'a eklemiyoruz.
LANGCHAIN_PROMPT = SYSTEM_PROMPT + """

Kullanıcı "bana uygun" gibi kişisel bir istekte bulunursa önce get_my_profile ile
profilini oku, sonra hedef rollerine göre search_jobs ile ara.
search_jobs'ta location parametresini kullanma; şehir ve remote uygunluğunu tercihlere göre sen değerlendir.
Tercihlerine uymayan ilanları (istemediği roller, uygun olmayan şehir/çalışma şekli) ele
ve neden elediğini tek cümleyle belirt.
İlanları listelerken id'lerini de göster ki kullanıcı "2'yi takibe al" diyebilsin.
Kullanıcı bir ilanı takibe almak isterse track_job, takip listesini sorarsa list_tracked_jobs kullan."""


def build_agent(checkpointer):
    """Agent'ı verilen checkpointer ile kurar. Checkpointer değişse de agent tanımı aynı kalır."""
    return create_agent(
        model=MODEL,
        tools=[get_my_profile, search_jobs, track_job, list_tracked_jobs],
        system_prompt=LANGCHAIN_PROMPT,
        context_schema=Context,
        state_schema=JobAgentState,
        checkpointer=checkpointer,
    )


def stream_agent(agent, question: str, context: Context, config: dict) -> None:
    """İki stream modunu aynı anda dinler:
    - "messages": modelin ürettiği her token (yazı canlı akar)
    - "updates":  her adım (node) bittiğinde o adımın sonucu (tool çağrıları, tool sonuçları)
    """
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        context=context,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata["langgraph_node"] == "model" and token.content:
                print(token.content, end="", flush=True)

        elif chunk["type"] == "updates":
            for node, update in chunk["data"].items():
                if not update or "messages" not in update:
                    continue
                msg = update["messages"][-1]
                if node == "model" and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n🔧 {tc['name']}({tc['args']})")
                elif node == "tools":
                    preview = msg.content[:80].replace("\n", " ")
                    print(f"   ↳ {msg.name}: {preview}...")
    print()


if __name__ == "__main__":
    # thread_id artık sabit: aynı isimle başlatınca kaldığın konuşmaya dönersin.
    # Yeni bir konuşma için: python -m langchain_agent.agent yeni-isim
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(resume=load_resume())

    # with bloğu: bağlantıyı açar, iş bitince (q ile çıkınca) düzgünce kapatır
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()  # checkpoint tablolarını oluşturur; varsa dokunmaz
        agent = build_agent(checkpointer)

        state = agent.get_state(config).values
        print(f"Thread: {thread_id} | kayıtlı mesaj: {len(state.get('messages', []))} | "
              f"takipte: {len(state.get('tracked_jobs', []))} ilan")
        print("Çıkmak için 'q' yaz. State'i görmek için '/state' yaz.\n")

        while True:
            question = input("Sen: ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if question == "/state":
                state = agent.get_state(config).values
                print(f"mesaj sayısı: {len(state.get('messages', []))}")
                print(f"tracked_jobs: {[j['title'] for j in state.get('tracked_jobs', [])]}\n")
                continue
            print("Agent: ", end="")
            stream_agent(agent, question, context, config)
            print()
"""LangGraph Adım 4: time travel.

Checkpointer her node'dan sonra state'in bir kopyasını (checkpoint) kaydediyor. Bu script:
  1. Bir thread'in checkpoint geçmişini listeler (get_state_history)
  2. Onay sorulan (interrupt) noktalardan birini seçtirir
  3. O noktadan yeni bir dal (fork) açar ve soruyu tekrar sorar

Kullanım: python -m langgraph_agent.time_travel adim3b
"""
import sys

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from langgraph_agent.graph import Context, build_graph
from langgraph_agent.main import stream_graph
from shared.config import DATABASE_URL
from shared.resume import load_resume


def last_user_message(snapshot) -> str:
    humans = [m for m in snapshot.values.get("messages", []) if m.type == "human"]
    return humans[-1].content[:40] if humans else "-"


def tracked_ids(snapshot) -> list[int]:
    return [j["id"] for j in snapshot.values.get("tracked_jobs", [])]


def time_travel(graph, config: dict, context: Context) -> None:
    # Yeniden eskiye sıralı geliyor; okunması kolay olsun diye çeviriyoruz
    history = list(graph.get_state_history(config))[::-1]

    print(f"\nCheckpoint geçmişi ({len(history)} adet):")
    print(f"{'step':>4}  {'sıradaki node':<16} {'takipte':<10} son kullanıcı mesajı")
    for s in history:
        nxt = ", ".join(s.next) or "(bitti)"
        mark = "  ✋" if s.interrupts else ""
        print(f"{s.metadata.get('step', '?'):>4}  {nxt:<16} {str(tracked_ids(s)):<10} {last_user_message(s)}{mark}")

    points = [s for s in history if s.next == ("confirm_track",) and s.interrupts]
    if not points:
        print("\nBu thread'de onay sorulan bir nokta yok.")
        return

    print("\nOnay sorulan noktalar:")
    for i, s in enumerate(points):
        print(f"  [{i}] step {s.metadata.get('step')}: {s.interrupts[0].value['jobs']} | o an takipte: {tracked_ids(s)}")
    while True:
        raw = input(f"Hangisine dönelim? Köşeli parantezdeki numarayı yaz (0-{len(points) - 1}): ").strip()
        if raw.isdigit() and int(raw) < len(points):
            break
        print("  Geçersiz seçim, sadece numara yaz (örn. 1).")
    target = points[int(raw)]

    # Neden doğrudan Command(resume=...) değil de fork?
    # O checkpoint, eski cevabı ("h") da kayıtlı tutuyor; doğrudan devam edersek eski cevap
    # tekrar kullanılır. update_state o noktadan YENİ bir checkpoint (dal) açar; eski cevap
    # bu dalda yok, bu yüzden interrupt soruyu yeniden sorar. Eski dal geçmişte durmaya devam eder.
    fork_config = graph.update_state(target.config, {}, as_node="router")

    print("\n↩️  O noktaya dönüldü, graph oradan tekrar çalışıyor:")
    stream_graph(graph, None, context, fork_config)  # None: yeni girdi yok, kaldığı yerden devam

    while interrupts := graph.get_state(config).interrupts:
        payload = interrupts[0].value
        print(f"{payload['question']}")
        for job in payload["jobs"]:
            print(f"  {job}")
        answer = input("(e/h): ").strip()
        stream_graph(graph, Command(resume=answer), context, config)

    head = graph.get_state(config)
    print(f"Şimdiki takip listesi: {tracked_ids(head)}")
    print(f"Toplam checkpoint: {len(history)} → {len(list(graph.get_state_history(config)))} "
          "(eski dal silinmedi, yenisi eklendi)")
    print("Not: seçtiğin noktadan SONRAKİ turlar bu yeni dalda yok; konuşma oradan devam ediyor.")


if __name__ == "__main__":
    thread_id = "lg-" + (sys.argv[1] if len(sys.argv) > 1 else "default")
    config = {"configurable": {"thread_id": thread_id}}
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        time_travel(build_graph(checkpointer), config, Context(resume=load_resume()))
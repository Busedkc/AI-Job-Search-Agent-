"""`langgraph dev` için giriş noktası.

langgraph dev kendi checkpointer'ını ve store'unu sağlıyor (geliştirme için, bellekte +
.langgraph_api klasöründe). Bu yüzden burada checkpointer/store VERMİYORUZ; Postgres'li
terminal versiyonu (main.py) olduğu gibi duruyor.
"""
from langgraph_agent.graph import build_graph

graph = build_graph()
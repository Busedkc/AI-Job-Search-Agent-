"""İki versiyonun da kullandığı ayarlar. .env repo kökünden okunur."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("MODEL", "groq:openai/gpt-oss-120b")
# Skorlama için ayrı model (boşsa MODEL kullanılır). Groq'ta her modelin rate limit'i ayrı.
SCORER_MODEL = os.getenv("SCORER_MODEL") or MODEL
DATA_DIR = ROOT_DIR / "shared" / "data"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://jobagent:jobagent@localhost:5433/jobagent")

# Store'da verinin kime ait olduğu (çok kullanıcılı bir uygulamada login'den gelirdi)
USER_ID = os.getenv("USER_ID", "me")
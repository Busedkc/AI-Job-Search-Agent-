"""İki versiyonun da kullandığı ayarlar. .env repo kökünden okunur."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MODEL = os.getenv("MODEL", "groq:openai/gpt-oss-120b")
DATA_DIR = ROOT_DIR / "shared" / "data"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://jobagent:jobagent@localhost:5433/jobagent")

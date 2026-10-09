"""CV okuma. Kişisel resume.md yoksa örnek dosyaya düşer."""
from shared.config import DATA_DIR


def load_resume() -> str:
    path = DATA_DIR / "resume.md"
    if not path.exists():
        path = DATA_DIR / "resume.example.md"
    return path.read_text(encoding="utf-8")
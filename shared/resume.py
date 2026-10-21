"""CV okuma. Kişisel resume.md yoksa örnek dosyaya düşer."""
from shared.config import DATA_DIR


def load_resume() -> str:
    path = DATA_DIR / "resume.md"
    if not path.exists():
        path = DATA_DIR / "resume.example.md"
    return path.read_text(encoding="utf-8")


def parse_target_roles(resume: str) -> list[str]:
    """CV'deki "Target roles:" satırını rollere böler. LLM gerektirmez."""
    for line in resume.splitlines():
        if "target roles" in line.lower():
            roles = line.split(":", 1)[1].replace("**", "")
            return [r.strip() for r in roles.split(",") if r.strip()]
    return []
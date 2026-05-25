"""Reads all skills/*/SKILL.md files and provides them as context."""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "skills"


def list_skill_docs() -> list[dict]:
    """Return [{"name": "news-digest", "content": "# News Digest\n..."}]."""
    results: list[dict] = []
    if SKILLS_DIR.exists():
        for subdir in sorted(SKILLS_DIR.iterdir()):
            skill_md = subdir / "SKILL.md"
            if skill_md.exists():
                results.append(
                    {
                        "name": subdir.name,
                        "content": skill_md.read_text(encoding="utf-8"),
                    }
                )
    return results


def get_skill_doc(name: str) -> str | None:
    """Read a specific SKILL.md by directory name."""
    path = SKILLS_DIR / name / "SKILL.md"
    return path.read_text(encoding="utf-8") if path.exists() else None

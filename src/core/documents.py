import asyncio
from pathlib import Path


_TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".log", ".json", ".yaml", ".yml"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks).strip()


def _read_docx(path: Path) -> str:
    import docx  # python-docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _extract_sync(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in _TEXT_EXTS:
        return _read_text(path)
    # неизвестный формат — пробуем как текст
    try:
        return _read_text(path)
    except Exception:
        return ""


async def extract_text(path: Path) -> str:
    """Извлекает текст из документа. Безопасно для больших файлов — в thread."""
    return await asyncio.to_thread(_extract_sync, path)


def is_supported(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in _TEXT_EXTS or suffix in {".pdf", ".docx"}

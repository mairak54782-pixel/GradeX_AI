"""Extract text from PDF, PPTX, DOCX, TXT files, then split it into overlapping chunks."""

from __future__ import annotations

from pathlib import Path

import pypdf
from docx import Document
from pptx import Presentation


def _extract_pdf(filepath: Path) -> str:
    text_parts = []
    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _extract_pptx(filepath: Path) -> str:
    text_parts = []
    prs = Presentation(filepath)
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        if cell.text:
                            text_parts.append(cell.text)
    return "\n".join(text_parts)


def _extract_docx(filepath: Path) -> str:
    text_parts = []
    doc = Document(filepath)
    for para in doc.paragraphs:
        text_parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text_parts.append(cell.text)
    return "\n".join(text_parts)


def _extract_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore")


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".ppt": _extract_pptx,
    ".pptx": _extract_pptx,
    ".doc": _extract_docx,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}


def extract_text(filepath: str | Path) -> str:
    """Return the raw text of a supported file, or '' if unsupported/empty."""
    path = Path(filepath)
    extractor = EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        return ""
    return extractor(path).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks so context isn't lost at chunk edges."""
    text = text.strip()
    if not text:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]

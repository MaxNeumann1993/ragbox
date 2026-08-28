"""Structure-aware chunking.

Instead of slicing text every N characters (which can cut a sentence or a
table row in half), we split along the document's own structure: headings
first, then paragraphs, and only fall back to a hard split for a single
paragraph that is larger than the target chunk size on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    heading_path: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Chunk text prefixed with its heading breadcrumb, for embedding/LLM context."""
        if not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}\n\n{self.text}"


def _split_into_sections(text: str) -> list[tuple[list[str], str]]:
    """Split markdown text into (heading_path, body) sections."""
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(([title for _, title in stack], body))
        body_lines.clear()

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)
    flush()

    if not sections and text.strip():
        sections = [([], text.strip())]
    return sections


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _hard_split(paragraph: str, chunk_size: int) -> list[str]:
    """Last resort for a single paragraph longer than chunk_size: split on
    sentence boundaries where possible, otherwise slice at chunk_size."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if len(sentences) == 1:
        return [paragraph[i : i + chunk_size] for i in range(0, len(paragraph), chunk_size)]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Chunk]:
    """Split text into overlapping, structure-aware chunks.

    Chunks are built by greedily combining whole paragraphs within a heading
    section up to `chunk_size` characters. `chunk_overlap` characters from the
    end of one chunk are carried into the start of the next so retrieval
    doesn't lose context at a chunk boundary.
    """
    chunks: list[Chunk] = []

    for heading_path, body in _split_into_sections(text):
        paragraphs: list[str] = []
        for paragraph in _split_paragraphs(body):
            if len(paragraph) > chunk_size:
                paragraphs.extend(_hard_split(paragraph, chunk_size))
            else:
                paragraphs.append(paragraph)

        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if current and len(candidate) > chunk_size:
                chunks.append(Chunk(text=current, heading_path=heading_path))
                overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
                current = f"{overlap_tail}\n\n{paragraph}".strip() if overlap_tail else paragraph
            else:
                current = candidate
        if current:
            chunks.append(Chunk(text=current, heading_path=heading_path))

    return chunks

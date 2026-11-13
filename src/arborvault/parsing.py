"""Deterministic, bounded parsers. Never fetches a source URI."""

import io
import re
import zipfile
from dataclasses import dataclass
from uuid import UUID, uuid5

from bs4 import BeautifulSoup
from docx import Document as Docx
from markdown_it import MarkdownIt
from pypdf import PdfReader

from .domain import Node

MAX_BYTES = 2_000_000
MAX_TEXT = 1_000_000
MAX_NODES = 2000
PART_SIZE = 4000


@dataclass
class Section:
    level: int
    title: str
    text: str
    locator: dict


def parse(data: bytes, filename: str) -> list[Section]:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Document must contain 1..2000000 bytes")
    ext = filename.rsplit(".", 1)[-1].lower()
    sections = [Section(0, "Preamble", "", {})]
    if ext in {"md", "markdown"}:
        text = data.decode("utf-8-sig")
        lines = text.splitlines(keepends=True)
        headings = [t for t in MarkdownIt().parse(text) if t.type == "heading_open"]
        start = 0
        for heading in headings:
            line, end = heading.map
            sections[-1].text += "".join(lines[start:line])
            title = re.sub(r"^#{1,6}\s+|\s+#+\s*$", "", lines[line].strip())
            sections.append(Section(int(heading.tag[1]), title, "", {"line": line + 1}))
            start = end
        sections[-1].text += "".join(lines[start:])
    elif ext in {"txt", "text"}:
        sections[0].text = data.decode("utf-8-sig")
    elif ext in {"html", "htm"}:
        soup = BeautifulSoup(data.decode("utf-8-sig"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Walk text once, avoiding duplicate content from nested containers.
        for item in soup.descendants:
            if getattr(item, "name", None) and re.fullmatch(r"h[1-6]", item.name):
                sections.append(Section(int(item.name[1]), item.get_text(" ", strip=True), "", {}))
            elif isinstance(item, str) and not item.find_parent(re.compile(r"^h[1-6]$")):
                sections[-1].text += str(item) + " "
    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted or len(reader.pages) > 200:
            raise ValueError("Encrypted PDFs and PDFs over 200 pages are unsupported")
        sections = []
        for i, page in enumerate(reader.pages):
            sections.append(Section(1, f"Page {i + 1}", page.extract_text() or "", {"page": i + 1}))
            if sum(len(s.text) for s in sections) > MAX_TEXT:
                raise ValueError("Extracted text exceeds limit")
    elif ext == "docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 20_000_000:
                raise ValueError("Expanded DOCX exceeds limit")
        document = Docx(io.BytesIO(data))
        for i, p in enumerate(document.paragraphs):
            match = re.fullmatch(r"Heading ([1-6])", p.style.name or "")
            if match:
                sections.append(Section(int(match[1]), p.text, "", {"paragraph": i + 1}))
            else:
                sections[-1].text += p.text + "\n"
        for table in document.tables:
            sections.append(
                Section(
                    1,
                    "Table",
                    "\n".join(" | ".join(c.text for c in row.cells) for row in table.rows),
                    {},
                )
            )
    else:
        raise ValueError("Supported formats: md, txt, html, pdf, docx")
    if sum(len(s.text) + len(s.title) for s in sections) > MAX_TEXT:
        raise ValueError("Extracted text exceeds limit")
    if any("\x00" in s.text or "\x00" in s.title for s in sections):
        raise ValueError("NUL characters are not valid document text")
    if not any(s.text.strip() for s in sections):
        raise ValueError("No extractable text; scanned PDFs need external OCR")
    if len(sections) > MAX_NODES:
        raise ValueError("Too many sections")
    return sections


def build_nodes(sections: list[Section], document_id: str, revision: int, title: str):
    nodes: list[Node] = []

    def add(parent, heading, content, locator):
        ordinal = len(nodes)
        identity = str(uuid5(UUID(document_id), f"{revision}:{ordinal}"))
        path = (parent.path if parent else []) + [heading]
        node = Node(
            identity, parent.id if parent else None, heading, content, path, ordinal, locator
        )
        nodes.append(node)
        return node

    root = add(None, title, "", {})
    stack = [(0, root)]
    for section in sections:
        text = section.text.strip()
        if section.level == 0:
            current = root
        else:
            while len(stack) > 1 and stack[-1][0] >= section.level:
                stack.pop()
            current = add(stack[-1][1], section.title[:500], "", section.locator)
            stack.append((section.level, current))
        current.content = text[:PART_SIZE]
        for offset in range(PART_SIZE, len(text), PART_SIZE):
            add(
                current,
                f"{current.title} (part {offset // PART_SIZE + 1})",
                text[offset : offset + PART_SIZE],
                {**section.locator, "character_offset": offset},
            )
    if len(nodes) > MAX_NODES:
        raise ValueError("Too many nodes")
    return nodes

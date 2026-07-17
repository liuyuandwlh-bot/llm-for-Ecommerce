"""
Chunker for financial documents - Round 2 rewrite.

Contract:
- ``chunk_document(doc)`` returns a ``ChunkingResult`` with separate
  ``children`` (indexed by retriever) and ``parents`` (for context expansion).
- Every child carries ``parent_chunk_id``; every parent carries
  ``child_ids``.
- Chunk IDs are stable SHA-256 over normalized text + key metadata so they
  don't depend on list position.
- Tables are preserved as their own chunks (markdown + bbox + headers).
- Max chunk size is enforced (overflow splits).
- Duplicate (doc_id, chunk_id) raises.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    page_start: int
    page_end: int
    section: str | None = None
    chunk_type: str = "text"  # text | table | mixed
    parent_chunk_id: str | None = None
    token_count: int = 0
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "parent_chunk_id": self.parent_chunk_id,
            "token_count": self.token_count,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }


@dataclass
class ChunkingResult:
    children: list[Chunk]
    parents: list[Chunk]

    @property
    def all_chunks(self) -> list[Chunk]:
        return list(self.children) + list(self.parents)


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: float = 0.15
    min_chunk_size: int = 50
    max_chunk_size: int = 1024
    merge_short_sections: bool = True
    table_as_chunk: bool = True
    parent_chunk_size: int = 2048


def _stable_id(*parts: Any) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def generate_chunk_id(
    doc_id: str,
    text: str,
    chunk_type: str,
    *,
    page_start: int = 0,
    page_end: int = 0,
    section: str | None = None,
    index: int = 0,
) -> str:
    norm = re.sub(r"\s+", " ", text or "").strip()
    return _stable_id(doc_id, chunk_type, section or "", page_start, page_end, index, norm[:120])


def generate_parent_id(
    doc_id: str,
    text: str,
    *,
    page_start: int = 0,
    page_end: int = 0,
    section: str | None = None,
    index: int = 0,
) -> str:
    norm = re.sub(r"\s+", " ", text or "").strip()
    return "p_" + _stable_id(
        doc_id, "parent", section or "", page_start, page_end, index, norm[:120]
    )


# ---------------------------------------------------------------------------
# Helpers for parsing structured blocks
# ---------------------------------------------------------------------------


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_to_markdown(table: Any) -> str:
    cells = getattr(table, "cells", None) or []
    if not cells:
        return str(getattr(table, "text", ""))
    lines: list[str] = []
    headers = getattr(table, "headers", None) or []
    if headers:
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    elif cells:
        first = cells[0]
        lines.append("| " + " | ".join(["---"] * len(first)) + " |")
    for row in cells:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - chinese
    return int(chinese / 1.5) + int(other / 4)


def _is_section_header(text: str) -> bool:
    if not text:
        return False
    if len(text) < 50 and any(
        text.startswith(p) for p in ["第", "一", "二", "三", "四", "五", "六", "七"]
    ):
        return True
    if 2 < len(text) < 50 and text.rstrip()[-1] in "：:":
        return True
    return False


# ---------------------------------------------------------------------------
# ChunkBuilder
# ---------------------------------------------------------------------------


class ChunkBuilder:
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        self._seen_ids: dict[str, int] = {}

    def _check_unique(self, chunk_id: str) -> None:
        if chunk_id in self._seen_ids:
            raise ValueError(f"duplicate chunk_id detected: {chunk_id}")
        self._seen_ids[chunk_id] = 1

    def chunk_document(self, doc: Any) -> ChunkingResult:
        children: list[Chunk] = []
        parents: list[Chunk] = []
        self._seen_ids = {}

        doc_id = getattr(doc, "doc_id", "doc")
        pages = list(getattr(doc, "pages", []) or [])

        if not pages:
            raise ValueError(f"document {doc_id!r} has no pages")

        for page in pages:
            children.extend(self._chunk_page(page, doc_id))

        for child in children:
            if not child.token_count:
                child.token_count = _estimate_tokens(child.text)

        parents = self._create_parents(children, doc_id)

        return ChunkingResult(children=children, parents=parents)

    def _chunk_page(self, page: Any, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        page_num = _to_int(getattr(page, "page_num", getattr(page, "page", 0)))

        # Tables first.
        if self.config.table_as_chunk:
            for table in getattr(page, "tables", []) or []:
                chunks.append(self._create_table_chunk(table, doc_id, page_num))

        text_blocks = [
            b
            for b in (getattr(page, "blocks", []) or [])
            if not getattr(b, "is_header", False)
            and not getattr(b, "is_footer", False)
            and (getattr(b, "text", "") or "").strip()
        ]

        buffer_texts: list[str] = []
        buffer_pages: list[int] = []
        current_section: str | None = None

        for block in text_blocks:
            block_page = _to_int(getattr(block, "page", page_num))
            text = getattr(block, "text", "") or ""
            if _is_section_header(text):
                current_section = text
            buffer_texts.append(text)
            buffer_pages.append(block_page)

            if _estimate_tokens("\n".join(buffer_texts)) >= self.config.chunk_size:
                chunk = self._make_text_chunk(
                    doc_id=doc_id,
                    texts=buffer_texts,
                    pages=buffer_pages,
                    section=current_section,
                    chunk_type="text",
                )
                chunks.append(chunk)
                if self.config.chunk_overlap > 0:
                    overlap = max(1, int(len(buffer_texts) * self.config.chunk_overlap))
                    buffer_texts = buffer_texts[-overlap:]
                    buffer_pages = buffer_pages[-overlap:]
                else:
                    buffer_texts = []
                    buffer_pages = []

        if buffer_texts:
            chunk = self._make_text_chunk(
                doc_id=doc_id,
                texts=buffer_texts,
                pages=buffer_pages,
                section=current_section,
                chunk_type="text",
            )
            chunks.append(chunk)

        return chunks

    def _make_text_chunk(
        self,
        doc_id: str,
        texts: Sequence[str],
        pages: Sequence[int],
        section: str | None,
        chunk_type: str,
        index: int = 0,
    ) -> Chunk:
        text = "\n".join(texts)
        page_start = min(pages) if pages else 0
        page_end = max(pages) if pages else 0
        chunk_id = generate_chunk_id(
            doc_id,
            text,
            chunk_type,
            page_start=page_start,
            page_end=page_end,
            section=section,
            index=index,
        )
        self._check_unique(chunk_id)
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=doc_id,
            page_start=page_start,
            page_end=page_end,
            section=section,
            chunk_type=chunk_type,
            token_count=_estimate_tokens(text),
            metadata={"source": "page_chunking", "section": section},
        )

    def _create_table_chunk(self, table: Any, doc_id: str, page_num: int) -> Chunk:
        markdown = _table_to_markdown(table)
        bbox = getattr(table, "bbox", None)
        headers = getattr(table, "headers", []) or []
        cells = getattr(table, "cells", []) or []
        chunk_id = generate_chunk_id(
            doc_id,
            markdown,
            "table",
            page_start=page_num,
            page_end=page_num,
            section=None,
            index=0,
        )
        self._check_unique(chunk_id)
        return Chunk(
            chunk_id=chunk_id,
            text=markdown,
            doc_id=doc_id,
            page_start=page_num,
            page_end=page_num,
            chunk_type="table",
            bbox=bbox,
            token_count=_estimate_tokens(markdown),
            metadata={
                "table_headers": headers,
                "table_rows": len(cells),
                "source": "table_chunking",
            },
        )

    def _create_parents(self, children: list[Chunk], doc_id: str) -> list[Chunk]:
        if self.config.parent_chunk_size <= 0:
            return []

        parents: list[Chunk] = []
        current_children: list[Chunk] = []
        current_texts: list[str] = []
        current_pages: list[int] = []

        def flush() -> None:
            nonlocal current_children, current_texts, current_pages
            if not current_children:
                return
            combined = "\n\n".join(current_texts)
            page_start = min(current_pages)
            page_end = max(current_pages)
            parent_id = generate_parent_id(
                doc_id,
                combined,
                page_start=page_start,
                page_end=page_end,
                section=current_children[0].section,
                index=len(parents),
            )
            self._check_unique(parent_id)
            parent = Chunk(
                chunk_id=parent_id,
                text=combined,
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                section=current_children[0].section,
                chunk_type="mixed",
                token_count=_estimate_tokens(combined),
                metadata={
                    "child_ids": [c.chunk_id for c in current_children],
                    "num_children": len(current_children),
                    "source": "parent_chunking",
                },
            )
            parents.append(parent)
            current_children = []
            current_texts = []
            current_pages = []

        for child in children:
            if child.chunk_type == "table":
                flush()
                # Tables become their own parent
                table_parent_id = generate_parent_id(
                    doc_id,
                    child.text,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    section=child.section,
                    index=len(parents),
                )
                self._check_unique(table_parent_id)
                table_parent = Chunk(
                    chunk_id=table_parent_id,
                    text=child.text,
                    doc_id=doc_id,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    section=child.section,
                    chunk_type="table",
                    metadata={
                        "child_ids": [child.chunk_id],
                        "num_children": 1,
                        "source": "parent_chunking_table",
                    },
                )
                parents.append(table_parent)
                continue

            current_children.append(child)
            current_texts.append(child.text)
            current_pages.append(child.page_start)
            combined = "\n\n".join(current_texts)
            if _estimate_tokens(combined) >= self.config.parent_chunk_size:
                flush()

        flush()
        # Wire children -> parents
        for parent in parents:
            for child_id in parent.metadata.get("child_ids", []):
                for child in children:
                    if child.chunk_id == child_id and child.parent_chunk_id is None:
                        child.parent_chunk_id = parent.chunk_id
                        break
        return parents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def chunk_corpus(
    input_dir: str, output_path: str, config: ChunkConfig | None = None
) -> ChunkingResult:
    """Chunk all parsed JSON documents in ``input_dir`` and write JSONL."""
    builder = ChunkBuilder(config)
    all_children: list[Chunk] = []
    all_parents: list[Chunk] = []
    seen_ids: dict[str, int] = {}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input dir does not exist: {input_dir}")

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"no parsed documents found in {input_dir}")

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        class _Block:
            def __init__(self, d: dict[str, Any]):
                self.text = d.get("text", "") or ""
                self.is_header = d.get("is_header", False)
                self.is_footer = d.get("is_footer", False)
                self.page = d.get("page", d.get("page_num", 0))

        class _Table:
            def __init__(self, d: dict[str, Any]):
                self.headers = d.get("headers", []) or []
                self.cells = d.get("cells", []) or []
                self.bbox = d.get("bbox")
                self.page = d.get("page", d.get("page_num", 0))

        class _Page:
            def __init__(self, d: dict[str, Any]):
                self.page_num = d.get("page_num", d.get("page", 0))
                self.blocks = [_Block(b) for b in d.get("blocks", [])]
                self.tables = [_Table(t) for t in d.get("tables", [])]

        class _Doc:
            def __init__(self, d: dict[str, Any], pages: list[_Page], fallback_id: str):
                self.doc_id = d.get("doc_id", fallback_id)
                self.title = d.get("title", "")
                self.pages = pages
                self.metadata = d.get("metadata", {})

        pages = [_Page(p) for p in data.get("pages", [])]
        doc = _Doc(data, pages, json_file.stem)
        result = builder.chunk_document(doc)
        for c in result.children:
            if c.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id across docs: {c.chunk_id}")
            seen_ids[c.chunk_id] = 1
            all_children.append(c)
        for p in result.parents:
            if p.chunk_id in seen_ids:
                raise ValueError(f"duplicate parent chunk_id: {p.chunk_id}")
            seen_ids[p.chunk_id] = 1
            all_parents.append(p)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_children:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        for p in all_parents:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    return ChunkingResult(children=all_children, parents=all_parents)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Chunk parsed finance documents")
    parser.add_argument("--input", required=True, help="Directory of parsed JSON documents")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-chunk-size", type=int, default=1024)
    parser.add_argument("--parent-chunk-size", type=int, default=2048)
    args = parser.parse_args()

    config = ChunkConfig(
        max_chunk_size=args.max_chunk_size, parent_chunk_size=args.parent_chunk_size
    )
    try:
        result = chunk_corpus(args.input, args.output, config=config)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    print(
        f"Wrote {len(result.children)} children and {len(result.parents)} parents to {args.output}"
    )
    return 0


if __name__ == "__main__":
    exit(main())

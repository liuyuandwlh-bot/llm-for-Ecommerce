"""
Chunker for Financial Documents

Based on recommended plan:
- Structural-aware chunking (not fixed-size)
- Parent-child chunks
- Table protection and dual representation
- Preserve page/bbox for citation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json

from .pdf_parser import ParsedDocument, ParsedPage, Table


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    text: str
    doc_id: str
    page_start: int
    page_end: int
    section: Optional[str]
    chunk_type: Literal["text", "table", "mixed"]
    parent_chunk_id: Optional[str] = None
    token_count: int = 0
    bbox: Optional[tuple] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
class ChunkConfig:
    """Configuration for chunking strategy."""
    chunk_size: int = 512  # tokens
    chunk_overlap: float = 0.15  # 15% overlap
    min_chunk_size: int = 50  # minimum tokens
    max_chunk_size: int = 1024  # maximum tokens
    merge_short_sections: bool = True
    table_as_chunk: bool = True  # Keep tables as separate chunks
    parent_chunk_size: int = 2048  # Parent chunk size


class ChunkBuilder:
    """
    Build chunks from parsed documents.

    Strategies:
    1. Fixed-size: Simple but loses structure
    2. Structural: By headers/sections (recommended)
    3. Parent-child: Small chunks for retrieval, large for context
    """

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()

    def chunk_document(self, doc: ParsedDocument) -> List[Chunk]:
        """Chunk a single document."""
        chunks = []

        # Process pages
        for page in doc.pages:
            page_chunks = self._chunk_page(page, doc.doc_id)
            chunks.extend(page_chunks)

        # Group into parent chunks
        if self.config.parent_chunk_size > 0:
            chunks = self._create_parent_chunks(chunks, doc.doc_id)

        # Assign token counts
        for chunk in chunks:
            chunk.token_count = self._estimate_tokens(chunk.text)

        return chunks

    def _chunk_page(self, page: ParsedPage, doc_id: str) -> List[Chunk]:
        """Chunk a single page."""
        chunks = []

        # Get non-header/footer blocks
        text_blocks = [
            b for b in page.blocks
            if not b.is_header and not b.is_footer and b.text.strip()
        ]

        # Process tables first
        if self.config.table_as_chunk:
            for table in page.tables:
                table_chunk = self._create_table_chunk(table, doc_id)
                chunks.append(table_chunk)

        # Process text blocks
        current_section = None
        current_texts = []
        current_pages = set()

        for block in text_blocks:
            # Update section if detected
            if self._is_section_header(block):
                current_section = block.text

            current_texts.append(block.text)
            current_pages.add(block.page)

            # Check if we should create a chunk
            combined_text = "\n".join(current_texts)
            estimated_tokens = self._estimate_tokens(combined_text)

            if estimated_tokens >= self.config.chunk_size:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    text=combined_text,
                    doc_id=doc_id,
                    page_start=min(current_pages),
                    page_end=max(current_pages),
                    section=current_section,
                    chunk_type="text",
                    metadata={"source": "page_chunking"},
                )
                chunks.append(chunk)

                # Keep overlap
                if self.config.chunk_overlap > 0:
                    overlap_lines = int(len(current_texts) * self.config.chunk_overlap)
                    current_texts = current_texts[-overlap_lines:] if overlap_lines > 0 else []
                else:
                    current_texts = []
                current_pages = set([block.page])

        # Don't forget remaining text
        if current_texts:
            combined_text = "\n".join(current_texts)
            if self._estimate_tokens(combined_text) >= self.config.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    text=combined_text,
                    doc_id=doc_id,
                    page_start=min(current_pages) if current_pages else page.page_num,
                    page_end=max(current_pages) if current_pages else page.page_num,
                    section=current_section,
                    chunk_type="text",
                    metadata={"source": "page_chunking"},
                )
                chunks.append(chunk)

        return chunks

    def _create_table_chunk(self, table: Table, doc_id: str) -> Chunk:
        """Create a chunk from a table."""
        # Create both structured and natural language versions
        markdown = table.to_markdown()

        return Chunk(
            chunk_id=f"{doc_id}_table_{table.page}_{hash(table.markdown) % 10000}",
            text=markdown,
            doc_id=doc_id,
            page_start=table.page,
            page_end=table.page,
            section=None,
            chunk_type="table",
            bbox=table.bbox,
            metadata={
                "table_headers": table.headers,
                "table_rows": len(table.cells),
            },
        )

    def _create_parent_chunks(self, chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        """Create parent chunks that encompass multiple child chunks."""
        parent_chunks = []
        current_children = []
        current_texts = []
        current_pages = set()

        for chunk in chunks:
            if chunk.chunk_type == "table":
                # Tables are their own parents
                if current_children:
                    parent = self._make_parent_chunk(
                        current_children, current_texts, doc_id
                    )
                    parent_chunks.append(parent)
                    current_children = []
                    current_texts = []

                # Add table as parent chunk
                parent = Chunk(
                    chunk_id=f"{doc_id}_parent_{len(parent_chunks)}",
                    text=chunk.text,
                    doc_id=doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    chunk_type="table",
                    metadata={"child_ids": [chunk.chunk_id]},
                )
                parent_chunks.append(parent)
            else:
                current_children.append(chunk)
                current_texts.append(chunk.text)
                current_pages.add(chunk.page)

                # Check if we should create parent
                combined = "\n".join(current_texts)
                if self._estimate_tokens(combined) >= self.config.parent_chunk_size:
                    parent = self._make_parent_chunk(
                        current_children, current_texts, doc_id
                    )
                    parent_chunks.append(parent)
                    current_children = []
                    current_texts = []
                    current_pages = set()

        # Handle remaining
        if current_children:
            parent = self._make_parent_chunk(current_children, current_texts, doc_id)
            parent_chunks.append(parent)

        return parent_chunks

    def _make_parent_chunk(
        self,
        children: List[Chunk],
        texts: List[str],
        doc_id: str,
    ) -> Chunk:
        """Create a parent chunk from children."""
        combined_text = "\n\n".join(texts)

        return Chunk(
            chunk_id=f"{doc_id}_parent_{hash(combined_text[:100]) % 100000}",
            text=combined_text,
            doc_id=doc_id,
            page_start=min(c.page_start for c in children),
            page_end=max(c.page_end for c in children),
            section=children[0].section if children else None,
            chunk_type="mixed",
            metadata={"child_ids": [c.chunk_id for c in children]},
        )

    def _is_section_header(self, block) -> bool:
        """Detect if a block is a section header."""
        # Simple heuristic: short text with large font or numbered pattern
        text = block.text.strip()

        # Numbered sections: "1.2.3", "第一节", etc.
        if any(text.startswith(p) for p in ["第", "一", "二", "三", "四", "五"]):
            return True

        # Short bold/header-like text
        if len(text) < 50 and text[-1] in "：:":
            return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~1.5 chars per token for Chinese)."""
        # Rough estimate: 1 token ≈ 1.5 Chinese characters or 4 English words
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars

        return int(chinese_chars / 1.5) + int(other_chars / 4)

    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')

        print(f"Saved {len(chunks)} chunks to {output_path}")


def chunk_corpus(input_dir: str, output_path: str, config: ChunkConfig = None):
    """Chunk a corpus of parsed documents."""
    from .pdf_parser import ParsedDocument

    builder = ChunkBuilder(config)
    all_chunks = []

    input_path = Path(input_dir)
    for json_file in input_path.glob("*.json"):
        print(f"Chunking: {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc = ParsedDocument(
            doc_id=data["doc_id"],
            title=data["title"],
            pages=[
                type('ParsedPage', (), {
                    'page_num': p['page_num'],
                    'blocks': [
                        type('TextBlock', (), b)()
                        for b in p['blocks']
                    ],
                    'tables': [
                        type('Table', (), {
                            'cells': t['cells'],
                            'page': t['page'],
                            'headers': t['headers'],
                            'markdown': t.get('markdown', ''),
                            'bbox': (0, 0, 0, 0),
                        })()
                        for t in p.get('tables', [])
                    ],
                })()
                for p in data.get('pages', [])
            ],
            metadata=data.get('metadata', {}),
        )

        chunks = builder.chunk_document(doc)
        all_chunks.extend(chunks)

    builder.save_chunks(all_chunks, output_path)
    return all_chunks


if __name__ == "__main__":
    print("Chunker Module")
    print("Usage: python -m src.finance_rag.chunker <parsed_dir> <output_path>")

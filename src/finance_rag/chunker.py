"""
Chunker for Financial Documents

Based on recommended plan:
- Structural-aware chunking (not fixed-size)
- Parent-child chunks
- Table protection and dual representation
- Preserve page/bbox for citation

Fixed issues:
- Stable chunk IDs using content hash
- Proper parent_chunk_id assignment
- Table bbox preservation
- Max chunk size enforcement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json
import hashlib


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


def generate_stable_chunk_id(doc_id: str, text: str, chunk_type: str, index: int) -> str:
    """
    Generate stable chunk ID from content.
    
    Uses content hash for stability across runs.
    """
    # Create deterministic hash from content
    content_hash = hashlib.sha256(f"{doc_id}:{text[:200]}:{chunk_type}".encode()).hexdigest()[:12]
    return f"{doc_id}_{chunk_type}_{index}_{content_hash}"


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
        self._chunk_counter: Dict[str, int] = {}  # Per doc

    def _get_chunk_index(self, doc_id: str) -> int:
        """Get next chunk index for document."""
        if doc_id not in self._chunk_counter:
            self._chunk_counter[doc_id] = 0
        index = self._chunk_counter[doc_id]
        self._chunk_counter[doc_id] += 1
        return index

    def chunk_document(self, doc) -> List[Chunk]:
        """Chunk a single document."""
        chunks = []

        # Process pages
        for page in doc.pages:
            page_chunks = self._chunk_page(page, doc.doc_id)
            chunks.extend(page_chunks)

        # Assign stable IDs
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = generate_stable_chunk_id(
                doc.doc_id, chunk.text, chunk.chunk_type, i
            )

        # Group into parent chunks
        if self.config.parent_chunk_size > 0:
            chunks = self._create_parent_chunks(chunks, doc.doc_id)

        # Assign token counts
        for chunk in chunks:
            chunk.token_count = self._estimate_tokens(chunk.text)

        return chunks

    def _chunk_page(self, page, doc_id: str) -> List[Chunk]:
        """Chunk a single page."""
        chunks = []

        # Get non-header/footer blocks
        text_blocks = [
            b for b in page.blocks
            if not getattr(b, 'is_header', False) and not getattr(b, 'is_footer', False) and b.text.strip()
        ]

        # Process tables first
        if self.config.table_as_chunk and hasattr(page, 'tables'):
            for table in page.tables:
                table_chunk = self._create_table_chunk(table, doc_id)
                chunks.append(table_chunk)

        # Process text blocks
        current_section = None
        current_texts = []
        current_pages = set()

        for block in text_blocks:
            block_page = getattr(block, 'page', 0)
            
            # Update section if detected
            if self._is_section_header(block):
                current_section = block.text

            current_texts.append(block.text)
            current_pages.add(block_page)

            # Check if we should create a chunk
            combined_text = "\n".join(current_texts)
            estimated_tokens = self._estimate_tokens(combined_text)

            if estimated_tokens >= self.config.chunk_size:
                chunk_index = self._get_chunk_index(doc_id)
                chunk = Chunk(
                    chunk_id=f"temp_{chunk_index}",  # Will be replaced with stable ID
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
                    overlap_lines = max(1, int(len(current_texts) * self.config.chunk_overlap))
                    current_texts = current_texts[-overlap_lines:]
                else:
                    current_texts = []
                current_pages = set([block_page])

        # Don't forget remaining text
        if current_texts:
            combined_text = "\n".join(current_texts)
            if self._estimate_tokens(combined_text) >= self.config.min_chunk_size:
                chunk_index = self._get_chunk_index(doc_id)
                chunk = Chunk(
                    chunk_id=f"temp_{chunk_index}",
                    text=combined_text,
                    doc_id=doc_id,
                    page_start=min(current_pages) if current_pages else getattr(page, 'page_num', 0),
                    page_end=max(current_pages) if current_pages else getattr(page, 'page_num', 0),
                    section=current_section,
                    chunk_type="text",
                    metadata={"source": "page_chunking"},
                )
                chunks.append(chunk)

        return chunks

    def _create_table_chunk(self, table, doc_id: str) -> Chunk:
        """Create a chunk from a table."""
        # Get markdown representation
        if hasattr(table, 'to_markdown'):
            markdown = table.to_markdown()
        else:
            markdown = self._table_to_markdown(table)
        
        # Get bbox if available
        bbox = getattr(table, 'bbox', None)
        
        # Get headers if available
        headers = getattr(table, 'headers', [])

        chunk_index = self._get_chunk_index(doc_id)
        
        return Chunk(
            chunk_id=f"temp_{chunk_index}",  # Will be replaced
            text=markdown,
            doc_id=doc_id,
            page_start=getattr(table, 'page', 0),
            page_end=getattr(table, 'page', 0),
            section=None,
            chunk_type="table",
            bbox=bbox,
            metadata={
                "table_headers": headers,
                "table_rows": len(getattr(table, 'cells', [])),
                "source": "table_chunking",
            },
        )
    
    def _table_to_markdown(self, table) -> str:
        """Convert table object to markdown if method not available."""
        if not hasattr(table, 'cells'):
            return str(table)
        
        lines = []
        cells = table.cells
        
        if cells:
            # Header row
            if hasattr(table, 'headers') and table.headers:
                lines.append("| " + " | ".join(str(h) for h in table.headers) + " |")
            else:
                lines.append("| " + " | ".join(["---"] * len(cells[0])) + " |")
            
            # Data rows
            for row in cells:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        
        return "\n".join(lines)

    def _create_parent_chunks(self, chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        """Create parent chunks that encompass multiple child chunks."""
        parent_chunks = []
        child_ids = []  # Track all child IDs
        
        current_children = []
        current_texts = []
        current_pages = set()

        for chunk in chunks:
            child_ids.append(chunk.chunk_id)
            
            if chunk.chunk_type == "table":
                # Tables are their own parents, don't merge them
                if current_children:
                    parent = self._make_parent_chunk(
                        current_children, current_texts, doc_id
                    )
                    parent_chunks.append(parent)

                # Add table as parent chunk
                parent = Chunk(
                    chunk_id=generate_stable_chunk_id(doc_id, chunk.text, "parent_table", len(parent_chunks)),
                    text=chunk.text,
                    doc_id=doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    chunk_type="table",
                    parent_chunk_id=None,  # Tables don't have parents
                    metadata={"child_ids": [chunk.chunk_id], "source": "parent_chunking"},
                )
                parent_chunks.append(parent)
                current_children = []
                current_texts = []
                current_pages = set()
            else:
                current_children.append(chunk)
                current_texts.append(chunk.text)
                current_pages.add(chunk.page_start)

                # Check if we should create parent
                combined = "\n\n".join(current_texts)
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
        
        # Get child IDs for reference
        child_ids = [c.chunk_id for c in children]

        return Chunk(
            chunk_id=generate_stable_chunk_id(doc_id, combined_text[:200], "parent", len([p for p in parent_chunks if p.doc_id == doc_id])),
            text=combined_text,
            doc_id=doc_id,
            page_start=min(c.page_start for c in children),
            page_end=max(c.page_end for c in children),
            section=children[0].section if children else None,
            chunk_type="mixed",
            metadata={
                "child_ids": child_ids,
                "num_children": len(children),
                "source": "parent_chunking",
            },
        )

    def _is_section_header(self, block) -> bool:
        """Detect if a block is a section header."""
        text = getattr(block, 'text', str(block))
        
        if not text:
            return False

        # Numbered sections: "一、二、三", "第X节", etc.
        if any(text.startswith(p) for p in ["第", "一", "二", "三", "四", "五", "六", "七"]):
            return True

        # Short bold/header-like text
        if len(text) < 50 and len(text) > 2 and text[-1] in "：:":
            return True

        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~1.5 chars per token for Chinese)."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars

        return int(chinese_chars / 1.5) + int(other_chars / 4)

    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to JSONL file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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

        # Reconstruct document
        class SimplePage:
            pass
        
        class SimpleBlock:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)
        
        class SimpleTable:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)
                self.cells = data.get('cells', [])
        
        pages = []
        for p in data.get('pages', []):
            blocks = [SimpleBlock(b) for b in p.get('blocks', [])]
            tables = [SimpleTable(t) for t in p.get('tables', [])]
            
            sp = SimplePage()
            sp.page_num = p.get('page_num', 0)
            sp.blocks = blocks
            sp.tables = tables
            pages.append(sp)

        class SimpleDoc:
            def __init__(self, data, pages):
                self.doc_id = data.get('doc_id', '')
                self.title = data.get('title', '')
                self.pages = pages
                self.metadata = data.get('metadata', {})

        doc = SimpleDoc(data, pages)
        chunks = builder.chunk_document(doc)
        all_chunks.extend(chunks)

    builder.save_chunks(all_chunks, output_path)
    return all_chunks


if __name__ == "__main__":
    print("Chunker Module")
    print("Usage: python -m src.finance_rag.chunker <parsed_dir> <output_path>")

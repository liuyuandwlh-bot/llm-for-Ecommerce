"""
PDF Parser for Financial Documents

Based on recommended plan:
- Tiered parsing: Docling > pdfplumber > PyMuPDF > PaddleOCR
- Preserve layout: headers, footers, columns, tables
- Page-level tracking for citations
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextBlock:
    """A block of text with position information."""
    text: str
    page: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font: str | None = None
    font_size: float | None = None
    is_header: bool = False
    is_footer: bool = False
    section: str | None = None


@dataclass
class TableCell:
    """A table cell."""
    text: str
    row: int
    col: int
    bbox: tuple[float, float, float, float]
    is_header: bool = False


@dataclass
class Table:
    """A table with cells."""
    cells: list[list[str]]
    page: int
    bbox: tuple[float, float, float, float]
    headers: list[str]
    markdown: str  # Markdown representation

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        lines = []

        # Header
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Rows
        for row in self.cells:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


@dataclass
class ParsedPage:
    """A parsed page with text blocks and tables."""
    page_num: int
    blocks: list[TextBlock]
    tables: list[Table]
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """A fully parsed document."""
    doc_id: str
    title: str
    pages: list[ParsedPage]
    metadata: dict = field(default_factory=dict)

    def get_all_text(self) -> str:
        """Get all text from document."""
        texts = []
        for page in self.pages:
            for block in page.blocks:
                if not block.is_header and not block.is_footer:
                    texts.append(block.text)
        return "\n\n".join(texts)

    def save(self, output_path: str):
        """Save parsed document to JSON."""
        data = {
            "doc_id": self.doc_id,
            "title": self.title,
            "metadata": self.metadata,
            "pages": [
                {
                    "page_num": p.page_num,
                    "blocks": [
                        {
                            "text": b.text,
                            "page": b.page,
                            "bbox": b.bbox,
                            "is_header": b.is_header,
                            "is_footer": b.is_footer,
                            "section": b.section,
                        }
                        for b in p.blocks
                    ],
                    "tables": [
                        {
                            "cells": t.cells,
                            "page": t.page,
                            "headers": t.headers,
                            "markdown": t.markdown,
                        }
                        for t in p.tables
                    ],
                }
                for p in self.pages
            ],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class PDFParser:
    """
    Financial document PDF parser.

    Supports multiple backends:
    1. Docling (MIT) - Best for complex layouts
    2. pdfplumber (MIT) - Good for tables
    3. PyMuPDF (AGPL/Commercial) - Fast, basic
    4. PaddleOCR - For scanned documents
    """

    def __init__(self, parser_type: str = "pdfplumber"):
        self.parser_type = parser_type

    def parse_document(
        self,
        pdf_path: Path,
        doc_id: str,
        title: str | None = None,
    ) -> ParsedDocument:
        """Parse a PDF document."""
        if self.parser_type == "pdfplumber":
            return self._parse_with_pdfplumber(pdf_path, doc_id, title)
        elif self.parser_type == "pymupdf":
            return self._parse_with_pymupdf(pdf_path, doc_id, title)
        elif self.parser_type == "docling":
            return self._parse_with_docling(pdf_path, doc_id, title)
        else:
            raise ValueError(f"Unknown parser type: {self.parser_type}")

    def _parse_with_pdfplumber(
        self,
        pdf_path: Path,
        doc_id: str,
        title: str | None = None,
    ) -> ParsedDocument:
        """Parse with pdfplumber (good for tables)."""
        import pdfplumber

        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                blocks = []
                tables = []

                # Extract text with position
                words = page.extract_words()
                if words:
                    # Group words into blocks based on vertical proximity
                    current_block = []
                    last_y = None

                    for word in words:
                        if last_y is None or abs(word['top'] - last_y) < 15:
                            current_block.append(word)
                        else:
                            if current_block:
                                text = ' '.join(w['text'] for w in current_block)
                                bbox = (
                                    min(w['x0'] for w in current_block),
                                    min(w['top'] for w in current_block),
                                    max(w['x1'] for w in current_block),
                                    max(w['bottom'] for w in current_block),
                                )
                                blocks.append(TextBlock(
                                    text=text,
                                    page=page_num,
                                    bbox=bbox,
                                ))
                            current_block = [word]
                        last_y = word['top']

                    # Don't forget last block
                    if current_block:
                        text = ' '.join(w['text'] for w in current_block)
                        bbox = (
                            min(w['x0'] for w in current_block),
                            min(w['top'] for w in current_block),
                            max(w['x1'] for w in current_block),
                            max(w['bottom'] for w in current_block),
                        )
                        blocks.append(TextBlock(
                            text=text,
                            page=page_num,
                            bbox=bbox,
                        ))

                # Extract tables
                tables_found = page.extract_tables()
                if tables_found:
                    for table_data in tables_found:
                        if table_data:
                            table = Table(
                                cells=table_data,
                                page=page_num,
                                bbox=(0, 0, page.width, page.height),
                                headers=table_data[0] if table_data else [],
                                markdown="",
                            )
                            table.markdown = table.to_markdown()
                            tables.append(table)

                # Post-process: identify headers and footers
                blocks = self._identify_headers_footers(blocks, page_num, page.height)

                pages.append(ParsedPage(
                    page_num=page_num,
                    blocks=blocks,
                    tables=tables,
                ))

        return ParsedDocument(
            doc_id=doc_id,
            title=title or pdf_path.stem,
            pages=pages,
            metadata={"parser": "pdfplumber", "source": str(pdf_path)},
        )

    def _parse_with_pymupdf(
        self,
        pdf_path: Path,
        doc_id: str,
        title: str | None = None,
    ) -> ParsedDocument:
        """Parse with PyMuPDF (fast, basic)."""
        import fitz

        pages = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks_data = page.get_text("dict")

            blocks = []
            tables = []  # PyMuPDF doesn't extract tables by default

            for block in blocks_data.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span["text"].strip()
                            if text:
                                bbox = tuple(span["bbox"])
                                blocks.append(TextBlock(
                                    text=text,
                                    page=page_num,
                                    bbox=bbox,
                                    font=span.get("font"),
                                    font_size=span.get("size"),
                                ))

            # Identify headers/footers
            blocks = self._identify_headers_footers(blocks, page_num, page.rect.height)

            pages.append(ParsedPage(
                page_num=page_num,
                blocks=blocks,
                tables=tables,
            ))

        doc.close()

        return ParsedDocument(
            doc_id=doc_id,
            title=title or pdf_path.stem,
            pages=pages,
            metadata={"parser": "pymupdf", "source": str(pdf_path)},
        )

    def _parse_with_docling(
        self,
        pdf_path: Path,
        doc_id: str,
        title: str | None = None,
    ) -> ParsedDocument:
        """Parse with Docling (best for complex layouts)."""
        # Docling integration
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(pdf_path)

            # Convert to our format
            pages = []
            for page_num, page in enumerate(result.pages):
                blocks = []
                for element in page.elements:
                    if hasattr(element, 'text'):
                        blocks.append(TextBlock(
                            text=element.text,
                            page=page_num,
                            bbox=(0, 0, 0, 0),  # Docling format different
                        ))
                pages.append(ParsedPage(page_num=page_num, blocks=blocks, tables=[]))

            return ParsedDocument(
                doc_id=doc_id,
                title=title or pdf_path.stem,
                pages=pages,
                metadata={"parser": "docling", "source": str(pdf_path)},
            )

        except ImportError:
            print("Docling not installed, falling back to pdfplumber")
            return self._parse_with_pdfplumber(pdf_path, doc_id, title)

    def _identify_headers_footers(
        self,
        blocks: list[TextBlock],
        page_num: int,
        page_height: float,
    ) -> list[TextBlock]:
        """Identify and mark headers and footers."""
        header_threshold = 50  # pixels from top
        footer_threshold = 50  # pixels from bottom

        for block in blocks:
            y0 = block.bbox[1]

            if y0 < header_threshold:
                block.is_header = True
            elif page_height - y0 < footer_threshold:
                block.is_footer = True

        return blocks


def parse_batch(
    input_dir: str,
    output_dir: str,
    parser_type: str = "pdfplumber",
    doc_ids: list[str] | None = None,
) -> list[ParsedDocument]:
    """Parse a batch of PDF documents."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parser = PDFParser(parser_type=parser_type)

    documents = []
    pdf_files = list(input_path.glob("*.pdf"))

    for pdf_file in pdf_files:
        doc_id = pdf_file.stem
        if doc_ids and doc_id not in doc_ids:
            continue

        print(f"Parsing: {pdf_file.name}")

        try:
            doc = parser.parse_document(pdf_file, doc_id)
            doc_path = output_path / f"{doc_id}.json"
            doc.save(str(doc_path))
            documents.append(doc)
        except Exception as e:
            print(f"Error parsing {pdf_file.name}: {e}")

    print(f"Parsed {len(documents)} documents")
    return documents


if __name__ == "__main__":
    print("PDF Parser Module")
    print("Usage: python -m src.finance_rag.pdf_parser parse_batch <input_dir> <output_dir>")

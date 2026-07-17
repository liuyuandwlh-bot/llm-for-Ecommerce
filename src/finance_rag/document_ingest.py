"""
Document Ingestion for Financial RAG

Handles document manifest tracking and download from official sources.
Based on recommended plan:
- Sources: 巨潮资讯, 上交所, 深交所, 央行, 发改委
- Manifest tracks source_id, publisher, published_at, license, checksum
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class DocumentManifest:
    """Manifest entry for a single document."""

    doc_id: str
    company: str
    stock_code: str | None
    report_type: Literal[
        "annual_report", "half_year_report", "quarterly_report", "policy_document", "other"
    ]
    fiscal_period: str
    published_at: str
    source_url: str
    pdf_sha256: str
    parser: str
    parser_version: str
    ocr_used: bool
    page_count: int
    parse_quality: str
    license_note: str
    acquired_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "company": self.company,
            "stock_code": self.stock_code,
            "report_type": self.report_type,
            "fiscal_period": self.fiscal_period,
            "published_at": self.published_at,
            "source_url": self.source_url,
            "pdf_sha256": self.pdf_sha256,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "ocr_used": self.ocr_used,
            "page_count": self.page_count,
            "parse_quality": self.parse_quality,
            "license_note": self.license_note,
            "acquired_at": self.acquired_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentManifest":
        return cls(**data)


class DocumentIngestor:
    """
    Handles document ingestion from official sources.

    Document sources (from recommended plan):
    - 巨潮资讯 (cninfo.com.cn): Listed company disclosures
    - 上交所 (sse.com.cn): Shanghai Stock Exchange disclosures
    - 深交所 (szse.cn): Shenzhen Stock Exchange disclosures
    - 中国人民银行 (pbc.gov.cn): PBOC policy documents
    - 国家发改委 (ndrc.gov.cn): NDRC policy documents
    """

    def __init__(self, save_dir: str = "data/raw/reports"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: list[DocumentManifest] = []
        self._manifest_path = self.save_dir.parent / "document_manifest.json"

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_doc_id(
        self,
        source: str,
        stock_code: str | None,
        report_type: str,
        fiscal_period: str,
    ) -> str:
        """Generate unique document ID."""
        base = f"{source}_{stock_code or 'policy'}_{report_type}_{fiscal_period}"
        return base.replace(" ", "_").lower()

    def add_document(
        self,
        file_path: Path,
        company: str,
        stock_code: str | None = None,
        report_type: str = "other",
        fiscal_period: str = "",
        published_at: str = "",
        source_url: str = "",
        parser: str = "docling",
        ocr_used: bool = False,
        parse_quality: str = "pass",
        license_note: str = "official-disclosure",
        notes: str = "",
    ) -> DocumentManifest:
        """Add a document to the manifest."""
        # Compute hash
        sha256 = self._compute_sha256(file_path)

        # Generate ID
        doc_id = self._generate_doc_id(
            source=source_url.split("/")[2] if source_url else "local",
            stock_code=stock_code,
            report_type=report_type,
            fiscal_period=fiscal_period,
        )

        # Get page count (best-effort, may fail for non-PDF inputs)
        page_count = 0
        try:
            import fitz  # type: ignore

            with fitz.open(str(file_path)) as doc:
                page_count = len(doc)
        except Exception:
            page_count = 0

        manifest = DocumentManifest(
            doc_id=doc_id,
            company=company,
            stock_code=stock_code,
            report_type=report_type,
            fiscal_period=fiscal_period,
            published_at=published_at,
            source_url=source_url,
            pdf_sha256=sha256,
            parser=parser,
            parser_version="1.0.0",
            ocr_used=ocr_used,
            page_count=page_count,
            parse_quality=parse_quality,
            license_note=license_note,
            notes=notes,
        )

        self.manifest.append(manifest)
        return manifest

    def save_manifest(self):
        """Save manifest to JSON file."""
        manifest_data = [m.to_dict() for m in self.manifest]

        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

        print(f"Manifest saved to: {self._manifest_path}")
        print(f"Total documents: {len(self.manifest)}")

    def load_manifest(self):
        """Load manifest from JSON file."""
        if not self._manifest_path.exists():
            print(f"No manifest found at: {self._manifest_path}")
            return

        with open(self._manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.manifest = [DocumentManifest.from_dict(m) for m in manifest_data]
        print(f"Loaded {len(self.manifest)} documents from manifest")

    def get_documents_by_type(self, report_type: str) -> list[DocumentManifest]:
        """Get all documents of a specific type."""
        return [m for m in self.manifest if m.report_type == report_type]

    def get_documents_by_company(self, company: str) -> list[DocumentManifest]:
        """Get all documents from a specific company."""
        return [m for m in self.manifest if m.company == company]


def create_sample_manifest():
    """Create a sample manifest for testing."""
    ingestor = DocumentIngestor()

    # Sample entries representing official disclosures
    sample_docs = [
        {
            "company": "虚构科技股份有限公司",
            "stock_code": "000001",
            "report_type": "annual_report",
            "fiscal_period": "2025",
            "published_at": "2026-03-28",
            "source_url": "https://www.cninfo.com.cn/example",
            "license_note": "official-disclosure; local-research; no-redistribution",
        },
        {
            "company": "虚构新能源股份有限公司",
            "stock_code": "000002",
            "report_type": "annual_report",
            "fiscal_period": "2025",
            "published_at": "2026-04-15",
            "source_url": "https://www.sse.com.cn/example",
            "license_note": "official-disclosure; local-research; no-redistribution",
        },
    ]

    for doc in sample_docs:
        # In real usage, would have actual PDF files
        print(f"Would add: {doc['company']} {doc['fiscal_period']} {doc['report_type']}")

    return ingestor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document ingest manifest helper")
    parser.add_argument("--save-dir", default="data/raw/reports")
    parser.add_argument(
        "--emit-sample",
        action="store_true",
        help="Print a sample manifest describing how a real "
        "ingest run would look; no PDF download happens.",
    )
    args = parser.parse_args()

    if args.emit_sample:
        create_sample_manifest()
    else:
        print("Document ingest helper. Use --emit-sample to print a sample.")

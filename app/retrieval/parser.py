"""Research paper PDF parser extracting text, sections, and page numbers."""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pymupdf


@dataclass
class PaperCatalogEntry:
    """Metadata for canonical research papers in library."""

    arxiv_id: str
    version: str
    paper_id: str  # e.g. "1706.03762v7"
    title: str
    source_url: str
    pdf_filename: str


CANONICAL_PAPERS: Dict[str, PaperCatalogEntry] = {
    "1706.03762v7": PaperCatalogEntry(
        arxiv_id="1706.03762",
        version="v7",
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        source_url="https://arxiv.org/pdf/1706.03762v7.pdf",
        pdf_filename="1706.03762v7_Attention_Is_All_You_Need.pdf",
    ),
    "2410.00037v1": PaperCatalogEntry(
        arxiv_id="2410.00037",
        version="v1",
        paper_id="2410.00037v1",
        title="Moshi: a speech-text foundation model for real-time dialogue",
        source_url="https://arxiv.org/pdf/2410.00037v1.pdf",
        pdf_filename="2410.00037v1_Moshi.pdf",
    ),
    "2507.07955v2": PaperCatalogEntry(
        arxiv_id="2507.07955",
        version="v2",
        paper_id="2507.07955v2",
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        source_url="https://arxiv.org/pdf/2507.07955v2.pdf",
        pdf_filename="2507.07955v2_Dynamic_Chunking.pdf",
    ),
}


@dataclass
class ParsedSectionBlock:
    """Extracted text block within a specific section and page."""

    paper_id: str
    paper_title: str
    version: str
    source_url: str
    page_number: int
    section: str
    text: str


class PaperParser:
    """Extracts structured text, section headings, and page boundaries from PDFs."""

    # Heading detection patterns
    _HEADING_PATTERNS = [
        re.compile(r"^([0-9]+(?:\.[0-9]+)*)\s+([A-Z][A-Za-z0-9\s\-:,\(\)]+)$"),
        re.compile(r"^(Abstract|Introduction|Conclusion|Discussion|References|Acknowledgements|Appendix(?:\s+[A-Z0-9]+)?)$", re.IGNORECASE),
    ]

    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = data_dir

    def parse_paper(self, paper_id: str) -> List[ParsedSectionBlock]:
        """Parse all pages and sections from a canonical paper."""
        if paper_id not in CANONICAL_PAPERS:
            raise ValueError(f"Unknown paper_id: {paper_id}. Must be one of {list(CANONICAL_PAPERS.keys())}")

        entry = CANONICAL_PAPERS[paper_id]
        pdf_path = os.path.join(self.data_dir, entry.pdf_filename)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file for {paper_id} not found at {pdf_path}")

        doc = pymupdf.open(pdf_path)
        blocks: List[ParsedSectionBlock] = []

        current_section = "Abstract" if paper_id != "2410.00037v1" else "Title & Authors"

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text()

            # Process lines to track section transitions
            lines = page_text.splitlines()
            current_buffer: List[str] = []

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Check for two-line headings like "3.1\nEncoder and Decoder Stacks"
                detected_heading: Optional[str] = None
                if re.match(r"^[0-9]+(?:\.[0-9]+)*$", line) and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and next_line[0].isupper() and len(next_line) < 80:
                        detected_heading = f"{line} {next_line}"
                        i += 1  # Skip next line as well

                if not detected_heading:
                    for pattern in self._HEADING_PATTERNS:
                        match = pattern.match(line)
                        if match and len(line) < 80:
                            detected_heading = line
                            break

                if detected_heading:
                    # Flush current buffer to previous section
                    if current_buffer:
                        block_text = " ".join(current_buffer).strip()
                        if block_text:
                            blocks.append(
                                ParsedSectionBlock(
                                    paper_id=entry.paper_id,
                                    paper_title=entry.title,
                                    version=entry.version,
                                    source_url=entry.source_url,
                                    page_number=page_num,
                                    section=current_section,
                                    text=block_text,
                                )
                            )
                        current_buffer = []
                    current_section = detected_heading
                else:
                    current_buffer.append(line)

                i += 1

            if current_buffer:
                block_text = " ".join(current_buffer).strip()
                if block_text:
                    blocks.append(
                        ParsedSectionBlock(
                            paper_id=entry.paper_id,
                            paper_title=entry.title,
                            version=entry.version,
                            source_url=entry.source_url,
                            page_number=page_num,
                            section=current_section,
                            text=block_text,
                        )
                    )

        return blocks

    def parse_all_papers(self) -> Dict[str, List[ParsedSectionBlock]]:
        """Parse all three canonical papers in the library."""
        results = {}
        for paper_id in CANONICAL_PAPERS:
            results[paper_id] = self.parse_paper(paper_id)
        return results

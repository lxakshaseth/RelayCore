"""Configurable text chunking preserving document hierarchy and metadata."""

import re
from typing import List
from app.retrieval.models import DocumentChunk
from app.retrieval.parser import ParsedSectionBlock


class DocumentChunker:
    """Segments parsed research paper sections into configurable, overlapping chunks."""

    def __init__(self, chunk_size_words: int = 350, overlap_words: int = 50):
        if overlap_words >= chunk_size_words:
            raise ValueError(f"overlap_words ({overlap_words}) must be less than chunk_size_words ({chunk_size_words})")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while respecting abbreviations."""
        # Simple sentence boundary regex
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_section_block(self, block: ParsedSectionBlock, start_chunk_idx: int = 0) -> List[DocumentChunk]:
        """Chunk an individual section block into discrete DocumentChunk items."""
        words = block.text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = self.chunk_size_words - self.overlap_words
        chunk_idx = start_chunk_idx

        # If block is smaller than chunk size, return it as a single chunk
        if len(words) <= self.chunk_size_words:
            chunk_id = f"{block.paper_id}_p{block.page_number}_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    paper_id=block.paper_id,
                    paper_title=block.paper_title,
                    version=block.version,
                    section=block.section,
                    page_number=block.page_number,
                    content=block.text,
                    source_url=block.source_url,
                    word_count=len(words),
                )
            )
            return chunks

        # Sliding window over words
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)
            chunk_id = f"{block.paper_id}_p{block.page_number}_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    paper_id=block.paper_id,
                    paper_title=block.paper_title,
                    version=block.version,
                    section=block.section,
                    page_number=block.page_number,
                    content=chunk_text,
                    source_url=block.source_url,
                    word_count=len(chunk_words),
                )
            )
            chunk_idx += 1

            # Stop if we reached the end of the text
            if i + self.chunk_size_words >= len(words):
                break

        return chunks

    def chunk_all_blocks(self, blocks: List[ParsedSectionBlock]) -> List[DocumentChunk]:
        """Process an entire stream of parsed section blocks into a unified chunk repository."""
        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for block in blocks:
            section_chunks = self.chunk_section_block(block, start_chunk_idx=chunk_counter)
            all_chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)

        return all_chunks

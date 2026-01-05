from typing import List, Optional
import re


class TextSplitter:
    """
    Utility class for splitting text into chunks for embedding and processing.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap to preserve context.
        """
        if not text or len(text.strip()) == 0:
            return []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                if paragraph.strip():
                    chunks.append(paragraph.strip())
            else:
                # If paragraph is too long, split it into sentences
                sentences = self._split_into_sentences(paragraph)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= self.chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Start new chunk with some overlap if possible
                        current_chunk = self._get_overlap(current_chunk) + sentence

                if current_chunk:
                    chunks.append(current_chunk)

        # Filter out empty chunks and clean up
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        """
        # Split on sentence boundaries (., !, ?, etc.) followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, text: str) -> str:
        """
        Get the last few words from text for overlap.
        """
        words = text.split()
        if len(words) <= self.overlap // 5:  # Approximate overlap based on words
            return text
        overlap_words = words[-(self.overlap // 5):]
        overlap_text = ' '.join(overlap_words)
        return overlap_text + ' ... '


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    """
    if not text:
        return text

    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text)

    # Remove extra newlines
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)

    return cleaned.strip()


def extract_citations(text: str) -> List[str]:
    """
    Extract potential citations from text.
    """
    # Look for patterns like "Chapter X", "Section X.Y", "Page N", etc.
    citation_patterns = [
        r'Chapter\s+\d+[A-Z]?',
        r'Section\s+\d+\.\d+',
        r'Page\s+\d+',
        r'Figure\s+\d+\.\d+',
        r'Table\s+\d+\.\d+',
        r'Equation\s+\d+\.\d+',
    ]

    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        citations.extend(matches)

    return list(set(citations))  # Remove duplicates
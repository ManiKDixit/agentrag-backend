# backend/app/knowledge/ingestion.py
"""
INGESTION: Converting raw documents into searchable text chunks.

KEY CONCEPT — "Chunking"
========================
Why can't we just embed the entire document?
1. Embedding models have token limits (8191 for OpenAI)
2. Even if it fit, a single vector for a 50-page doc loses detail
3. We want to retrieve SPECIFIC relevant paragraphs, not the whole doc

Chunking strategy matters enormously for RAG quality:
- Too small (50 tokens): loses context, retrieves fragments
- Too large (2000 tokens): dilutes relevance, wastes context window
- Sweet spot: 300-600 tokens with 50-100 token overlap

The OVERLAP ensures we don't cut a concept in half at a chunk boundary.
"""
import io
from typing import List
from PyPDF2 import PdfReader
import tiktoken


class TextChunker:
    """
    Splits text into overlapping chunks of approximately `chunk_size` tokens.

    Uses tiktoken (OpenAI's tokeniser) to count tokens accurately.
    WHY tiktoken? Because "word count" ≠ token count.
    "unhappiness" = 3 tokens: ["un", "happiness", ... ] depending on the model.
    We need accurate token counts to stay within embedding model limits.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # cl100k_base is the tokeniser used by text-embedding-3-small and GPT-4
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """
        Algorithm:
        1. Split text into sentences (by newlines and periods)
        2. Greedily pack sentences into chunks until chunk_size is reached
        3. Start next chunk with `chunk_overlap` tokens from end of previous
        """
        # Split on paragraph breaks first, then sentences
        paragraphs = text.split("\n\n")
        sentences = []
        for para in paragraphs:
            # Keep paragraph structure as a hint
            for sent in para.split(". "):
                cleaned = sent.strip()
                if cleaned:
                    sentences.append(cleaned + ".")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Current chunk is full — save it
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # OVERLAP: keep the last N tokens worth of sentences
                # This ensures continuity between chunks
                overlap_chunk = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_tokens += s_tokens

                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def parse_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Routes to the correct parser based on file extension.
    This is easily extensible — add DOCX, HTML, etc. parsers here.
    """
    if filename.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif filename.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {filename}")
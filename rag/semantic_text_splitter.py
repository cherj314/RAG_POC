import re
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticTextSplitter(TextSplitter):
    """
    A text splitter that creates chunks based on semantic similarity while respecting document structure.
    It smartly splits text at natural boundaries like paragraphs, headings, and chapters.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        min_chunk_size: int = 200,
        max_chunk_size: int = 3000,
        chunk_overlap: int = 200,
        respect_structure: bool = True,
        verbose: bool = False
    ):
        super().__init__(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_structure = respect_structure
        self.verbose = verbose
        self.embedding_model = None  # Lazy loading
        
        # Patterns for structural elements
        self.patterns = {
            'chapter': re.compile(r'^(?:CHAPTER|Chapter)\s+[IVXLCDM\d]+', re.MULTILINE),
            'scene_break': re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE),
            'page_number': re.compile(r'^\s*\d+\s*$', re.MULTILINE),
            'header': re.compile(r'HARRY POTTER|^THE\s+[A-Z\s]+$', re.MULTILINE)
        }
    
    def _log(self, message: str) -> None:
        """Print log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticSplitter] {message}")
    
    def _get_embedding_model(self):
        """Lazy load the embedding model when needed."""
        if self.embedding_model is None:
            self._log(f"Loading embedding model: {self.embedding_model}")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.embedding_model
    
    def _preprocess_text(self, text: str) -> str:
        """Clean text by removing headers, footers, page numbers."""
        # Remove page numbers
        text = self.patterns['page_number'].sub('', text)
        
        # Remove headers
        text = self.patterns['header'].sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix quotes for sentence detection
        text = re.sub(r'([.!?])"(\s)', r'\1" \2', text)
        
        return text.strip()
    
    def _is_structural_boundary(self, text: str) -> bool:
        """Check if text contains a structural boundary like a chapter heading."""
        if not self.respect_structure:
            return False
            
        first_line = text.strip().split('\n')[0] if text.strip() else ""
        
        return (
            self.patterns['chapter'].search(first_line) or 
            self.patterns['scene_break'].search(first_line)
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text segments."""
        # For very short texts, consider them similar to avoid over-chunking
        if len(text1) < 100 or len(text2) < 100:
            return 1.0
        
        try:
            model = self._get_embedding_model()
            embed1 = model.encode(text1)
            embed2 = model.encode(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
            return float(similarity)
        except Exception as e:
            self._log(f"Error calculating similarity: {str(e)}")
            return 0.0  # Assume not similar if there's an error
    
    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        if not text or not text.strip():
            return []
        
        # Preprocess the text
        cleaned_text = self._preprocess_text(text)
        
        # Initial splitting into paragraphs
        paragraphs = re.split(r'\n\s*\n', cleaned_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Group paragraphs into chunks based on size and semantic similarity
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Always start a new chunk at chapter headings or scene breaks
            if self._is_structural_boundary(paragraph):
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = len(paragraph)
                continue
                
            # Check if adding this paragraph would exceed max size
            if current_size + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = len(paragraph)
                continue
                
            # If we have content in the current chunk, check semantic similarity
            if current_chunk:
                last_text = current_chunk[-1]
                # Only check similarity if both texts are substantial
                if len(last_text) > 100 and len(paragraph) > 100:
                    similarity = self._calculate_similarity(last_text, paragraph)
                    if similarity < self.similarity_threshold and current_size > self.min_chunk_size:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = [paragraph]
                        current_size = len(paragraph)
                        continue
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 4  # +4 for the "\n\n" separator
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        # Handle small chunks - merge with neighbors if below minimum size
        merged_chunks = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            
            # If this chunk is too small and not the last one, try to merge with next
            if len(current) < self.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                # Only merge if combined size doesn't exceed max
                if len(current) + len(next_chunk) <= self.max_chunk_size:
                    merged_chunks.append(current + "\n\n" + next_chunk)
                    i += 2  # Skip the next chunk since we merged it
                else:
                    merged_chunks.append(current)
                    i += 1
            else:
                merged_chunks.append(current)
                i += 1
        
        return merged_chunks
    
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Convert texts to Document objects with metadata."""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                # Add structure detection metadata
                if self.respect_structure:
                    lines = chunk.strip().split('\n')
                    if lines and self.patterns['chapter'].search(lines[0]):
                        doc_metadata["contains_chapter_heading"] = True
                        doc_metadata["current_chapter"] = lines[0].strip()
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.create_documents(texts, metadatas)
import re
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticTextSplitter(TextSplitter):
    """
    A text splitter that creates chunks based on semantic similarity while respecting document structure.
    It smartly splits text at natural boundaries like paragraphs, headings, chapters, and sentences.
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
            'header': re.compile(r'HARRY POTTER|^THE\s+[A-Z\s]+$', re.MULTILINE),
            # Pattern to identify sentence boundaries
            'sentence_end': re.compile(r'[.!?][\'\"]?\s+')
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
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex pattern matching."""
        if not text:
            return []
        
        # Split text at sentence boundaries (., !, ?)
        # This looks for ./?/! followed by optional quotation marks, then whitespace
        sentences = self.patterns['sentence_end'].split(text)
        
        # Add the sentence terminators back, except for the last empty element
        result = []
        terminators = re.findall(self.patterns['sentence_end'], text)
        
        for i, sentence in enumerate(sentences[:-1]):  # All but the last
            if i < len(terminators):
                result.append(sentence + terminators[i])
            else:
                result.append(sentence)
        
        # Add the last element if it's not empty
        if sentences[-1].strip():
            result.append(sentences[-1])
        
        return [s.strip() for s in result if s.strip()]
    
    def _adjust_chunk_boundary_to_sentence(self, chunk: str) -> str:
        """Ensure a chunk ends at a sentence boundary."""
        # If the chunk already ends with a sentence terminator, return it as is
        if re.search(r'[.!?][\'\"]?$', chunk.strip()):
            return chunk
        
        # Find the last sentence boundary
        match = list(re.finditer(r'[.!?][\'\"]?\s+', chunk))
        if match:
            # Get the position of the last sentence terminator
            last_terminator_pos = match[-1].end()
            # Return the chunk up to that position
            return chunk[:last_terminator_pos].strip()
        
        return chunk  # If no sentence boundary found, return as is
    
    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks, respecting sentence boundaries."""
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
                    chunk_text = "\n\n".join(current_chunk)
                    # Ensure chunk ends at a sentence boundary before adding to chunks
                    chunks.append(self._adjust_chunk_boundary_to_sentence(chunk_text))
                current_chunk = [paragraph]
                current_size = len(paragraph)
                continue
                
            # Check if adding this paragraph would exceed max size
            if current_size + len(paragraph) > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                # Ensure chunk ends at a sentence boundary
                chunks.append(self._adjust_chunk_boundary_to_sentence(chunk_text))
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
                        chunk_text = "\n\n".join(current_chunk)
                        # Ensure chunk ends at a sentence boundary
                        chunks.append(self._adjust_chunk_boundary_to_sentence(chunk_text))
                        current_chunk = [paragraph]
                        current_size = len(paragraph)
                        continue
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 4  # +4 for the "\n\n" separator
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            # Ensure chunk ends at a sentence boundary
            chunks.append(self._adjust_chunk_boundary_to_sentence(chunk_text))
        
        # Handle small chunks - merge with neighbors if below minimum size
        # Also ensure all chunks start and end at sentence boundaries
        final_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Skip very small chunks (they might be merged with next)
            if len(chunk) < self.min_chunk_size and i < len(chunks) - 1:
                # Try to merge with next chunk
                next_chunk = chunks[i + 1]
                combined = chunk + "\n\n" + next_chunk
                
                if len(combined) <= self.max_chunk_size:
                    # We'll handle this in the next iteration by skipping this chunk
                    continue
            
            # Further split chunks if they're too large and contain multiple sentences
            if len(chunk) > self.max_chunk_size:
                sentences = self._split_into_sentences(chunk)
                if len(sentences) > 1:
                    sub_chunks = []
                    current_sub_chunk = []
                    current_sub_size = 0
                    
                    for sentence in sentences:
                        if current_sub_size + len(sentence) > self.max_chunk_size and current_sub_chunk:
                            sub_chunks.append(" ".join(current_sub_chunk))
                            current_sub_chunk = [sentence]
                            current_sub_size = len(sentence)
                        else:
                            current_sub_chunk.append(sentence)
                            current_sub_size += len(sentence) + 1  # +1 for space
                    
                    if current_sub_chunk:
                        sub_chunks.append(" ".join(current_sub_chunk))
                    
                    final_chunks.extend(sub_chunks)
                else:
                    # If it's a single long sentence, keep it
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        # Ensure overlap between chunks to improve retrieval
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            overlapped_chunks = []
            
            for i in range(len(final_chunks)):
                if i == 0:
                    # First chunk remains as is
                    overlapped_chunks.append(final_chunks[i])
                else:
                    prev_chunk = final_chunks[i-1]
                    current_chunk = final_chunks[i]
                    
                    # Find sentences at the end of the previous chunk
                    prev_sentences = self._split_into_sentences(prev_chunk)
                    if len(prev_sentences) > 0:
                        # Take the last few sentences from previous chunk
                        overlap_size = 0
                        overlap_sentences = []
                        
                        for sentence in reversed(prev_sentences):
                            if overlap_size + len(sentence) > self.chunk_overlap:
                                break
                            overlap_sentences.insert(0, sentence)
                            overlap_size += len(sentence) + 1  # +1 for space
                        
                        if overlap_sentences:
                            # Add these sentences to the start of the current chunk
                            overlap_text = " ".join(overlap_sentences)
                            if not current_chunk.startswith(overlap_text):
                                current_chunk = overlap_text + " " + current_chunk
                    
                    overlapped_chunks.append(current_chunk)
            
            return overlapped_chunks
        
        return final_chunks
    
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
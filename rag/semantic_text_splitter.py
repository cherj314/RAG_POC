"""
Enhanced Semantic Text Splitter for RAGbot

This module provides a semantic text splitter that divides documents
based on structural elements and meaning rather than arbitrary character counts.
"""

import nltk
import torch
import re
from typing import List, Optional, Dict, Tuple, Any
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Download only the necessary NLTK data package instead of all data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SemanticTextSplitter(TextSplitter):
    """
    Split text based on document structure and semantic meaning rather than fixed-size chunks.
    Recognizes headings, sections, lists, and other structural elements.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50,
        paragraph_separator: str = "\n\n",
        sentence_separator: str = "\n",
        respect_structure: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the enhanced semantic text splitter.
        
        Args:
            embedding_model: The embedding model to use for semantic similarity
            similarity_threshold: Threshold for determining semantic similarity (0-1)
            min_chunk_size: Minimum size of chunks in characters
            max_chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            paragraph_separator: String that separates paragraphs
            sentence_separator: String that separates sentences within paragraphs
            respect_structure: Whether to respect document structure like headings and lists
            verbose: Whether to print verbose logs
        """
        super().__init__(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.sentence_separator = sentence_separator
        self.respect_structure = respect_structure
        self.verbose = verbose
        
        # Structural patterns for recognizing document elements
        self.patterns = self._compile_patterns()
        
        # Initialize the embedding model with proper device detection
        self._log(f"Initializing embedding model: {embedding_model}")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"Using device: {device}")
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        except Exception as e:
            self._log(f"Error initializing model with device detection: {str(e)}")
            self._log("Falling back to basic model initialization")
            self.embedding_model = SentenceTransformer(embedding_model)
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for identifying structural elements in text."""
        return {
            # Heading patterns
            'markdown_heading': re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            'numbered_heading': re.compile(r'^(\d+\.)+\s+[A-Z].*$', re.MULTILINE),
            'underlined_heading': re.compile(r'^[^\n]+\n[-=]+$', re.MULTILINE),
            'capitalized_heading': re.compile(r'^[A-Z][A-Z\s]+[A-Z]$', re.MULTILINE),
            
            # List patterns
            'bullet_point': re.compile(r'^\s*[-•*+]\s+.+$', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+.+$', re.MULTILINE),
            'letter_list': re.compile(r'^\s*[A-Z]\.\s+.+$', re.MULTILINE),
            
            # Section boundaries
            'horizontal_rule': re.compile(r'^[-*=_]{3,}$', re.MULTILINE),
            'page_break': re.compile(r'(\f|\n{4,})'),
            
            # Special structures
            'table_row': re.compile(r'^\|.+\|$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]+?```'),
            
            # PDF-specific patterns
            'page_number': re.compile(r'^\s*\d+\s*$', re.MULTILINE),
            'footer': re.compile(r'\n[^.!?]*(?:[Pp]age|[Cc]opyright|©|\([Cc]\))[^.!?]*$'),
            'header': re.compile(r'^[^.!?]*(?:[Cc]hapter|[Ss]ection|SECTION|CHAPTER)[^.!?]*\n'),
        }
    
    def _log(self, message: str) -> None:
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticTextSplitter] {message}")
    
    def _is_structural_boundary(self, text: str) -> bool:
        """Determine if the text contains a structural boundary."""
        if not self.respect_structure:
            return False
            
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(text):
                self._log(f"Found structural boundary: {pattern_name}")
                return True
        return False
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Handle zero vectors
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            return 0.0
        
        try:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            self._log(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """Determine if two text segments are semantically similar."""
        # For very short texts, consider them similar to avoid over-chunking
        if len(text1) < 50 or len(text2) < 50:
            return True
        
        try:
            # Get embeddings for both text segments
            embedding1 = self.embedding_model.encode(text1, convert_to_numpy=True)
            embedding2 = self.embedding_model.encode(text2, convert_to_numpy=True)
            
            # Calculate similarity
            similarity = self._calculate_similarity(embedding1, embedding2)
            self._log(f"Semantic similarity: {similarity:.4f} (threshold: {self.similarity_threshold})")
            
            return similarity >= self.similarity_threshold
        except Exception as e:
            self._log(f"Error in semantic similarity check: {str(e)}")
            return True
    
    def _identify_structural_elements(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify structural elements in the text.
        
        Returns a list of tuples (start_pos, end_pos, element_type) for each element.
        """
        elements = []
        
        # Skip if structure respect is disabled
        if not self.respect_structure:
            return elements
        
        # Check for each pattern
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                elements.append((match.start(), match.end(), pattern_name))
        
        # Sort by start position
        elements.sort(key=lambda x: x[0])
        return elements
    
    def _segment_with_structure(self, text: str) -> List[str]:
        """
        Segment text into chunks based on structural elements and semantic boundaries.
        This is the main enhanced algorithm that improves upon the basic paragraph splitting.
        Ensures no content is lost during chunking.
        """
        if not text.strip():
            return []
        
        self._log(f"Processing text with structure awareness: {len(text)} characters")
        
        # Identify structural elements
        structural_elements = self._identify_structural_elements(text)
        
        # If no structural elements found, fall back to basic splitting
        if not structural_elements and not self.respect_structure:
            return self._segment_text_basic(text)
        
        # Split text by paragraphs first
        paragraphs = text.split(self.paragraph_separator)
        
        # Keep track of original paragraph content, including empty ones
        # We'll use this to verify content preservation later
        original_paragraphs = [p for p in paragraphs]
        
        # Filter out empty paragraphs for processing
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            # If no non-empty paragraphs, return the original text
            return [text] if text.strip() else []
        
        # Process paragraphs with structure awareness
        chunks = []
        current_chunk = ""
        current_structure = None  # Track current structural context (e.g., inside a list)
        
        for para in paragraphs:
            # Check if this paragraph defines a structural boundary
            is_boundary = self._is_structural_boundary(para)
            
            # Start a new chunk if:
            # 1. We've exceeded max size or
            # 2. We've hit a structural boundary and have content or
            # 3. Semantic similarity drops below threshold
            should_split = (len(current_chunk) + len(para) > self.max_chunk_size) or \
                          (is_boundary and current_chunk) or \
                          (current_chunk and not self._is_semantically_similar(current_chunk, para))
            
            if should_split:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += self.paragraph_separator + para
            else:
                current_chunk = para
            
            # If this paragraph is a heading or important structural element, 
            # it gets special treatment - we ensure it starts a chunk
            is_heading = False
            for _, _, name in structural_elements:
                if name.endswith('heading') and name in para:
                    is_heading = True
                    break
            
            if is_heading and len(current_chunk) > len(para):
                chunks.append(current_chunk)
                current_chunk = ""
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # If we have no chunks at this point (unlikely), return the original text
        if not chunks:
            self._log("Warning: No chunks created during structure-based splitting")
            return [text]
        
        # Ensure chunks meet minimum size by merging small chunks
        merged_chunks = self._merge_small_chunks(chunks)
        
        # Double-check we didn't lose content
        original_content = ''.join(original_paragraphs).strip()
        merged_content = self.paragraph_separator.join(merged_chunks).strip()
        
        # If we lost significant content, fall back to the pre-merged chunks
        if len(merged_content) < len(original_content) * 0.9:
            self._log(f"Warning: Content loss detected during structure-based chunking")
            if len(''.join(chunks).strip()) >= len(original_content) * 0.9:
                return chunks
            else:
                # Ultimate fallback if even pre-merged chunks lost content
                return [text]
        
        return merged_chunks
    
    def _segment_text_basic(self, text: str) -> List[str]:
        """Fall back to basic segmentation without structural awareness."""
        if not text:
            return []
            
        self._log(f"Falling back to basic text splitting")
        
        # Split by paragraphs first
        paragraphs = text.split(self.paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [text] if text.strip() else []
        
        # Combine paragraphs to meet requirements
        chunks = []
        current_chunk = ""
        last_added_text = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split it into sentences
            if len(paragraph) > self.max_chunk_size:
                try:
                    sentences = nltk.sent_tokenize(paragraph)
                except Exception:
                    # Fallback to simple splitting
                    sentences = [s.strip() + "." for s in paragraph.split(". ") if s.strip()]
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # If adding this sentence would exceed max size, start a new chunk
                    if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Keep overlap for context
                            current_chunk = last_added_text if self.chunk_overlap > 0 else ""
                    
                    # Check semantic similarity if we have content
                    if current_chunk and not self._is_semantically_similar(current_chunk, sentence):
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Add the sentence to the current chunk
                    if current_chunk:
                        current_chunk += self.sentence_separator + sentence
                    else:
                        current_chunk = sentence
                    
                    last_added_text = sentence
            else:
                # For shorter paragraphs, process as a unit
                if len(current_chunk) + len(paragraph) + 1 > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = last_added_text if self.chunk_overlap > 0 else ""
                
                # Check semantic similarity if we have content
                if current_chunk and not self._is_semantically_similar(current_chunk, paragraph):
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Add the paragraph to the current chunk
                if current_chunk:
                    current_chunk += self.paragraph_separator + paragraph
                else:
                    current_chunk = paragraph
                
                last_added_text = paragraph
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return self._merge_small_chunks(chunks)
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Ensure all chunks meet minimum size by merging small chunks, never discarding content."""
        if not chunks:
            return []
            
        # If we only have one chunk, just return it regardless of size
        if len(chunks) == 1:
            return chunks
            
        final_chunks = []
        small_chunk = ""
        
        for chunk in chunks:
            # Skip truly empty chunks (after stripping)
            if not chunk.strip():
                continue
                
            # If this chunk is small, add it to our buffer of small chunks
            if len(chunk) < self.min_chunk_size:
                small_chunk += (self.paragraph_separator + chunk) if small_chunk else chunk
                
                # If we've accumulated enough small chunks, add them as one chunk
                if len(small_chunk) >= self.min_chunk_size:
                    final_chunks.append(small_chunk)
                    small_chunk = ""
            else:
                # This is a normal-sized chunk
                final_chunks.append(chunk)
        
        # Handle any remaining small chunks - never discard them
        if small_chunk:
            if final_chunks:
                # Try to merge with the last chunk if it won't make it too large
                if len(final_chunks[-1]) + len(small_chunk) + len(self.paragraph_separator) <= self.max_chunk_size:
                    final_chunks[-1] += self.paragraph_separator + small_chunk
                else:
                    # If it would make the last chunk too large, keep it separate
                    final_chunks.append(small_chunk)
            else:
                # If there are no other chunks, keep this one even if it's small
                final_chunks.append(small_chunk)
        
        self._log(f"Created {len(final_chunks)} chunks after merging small chunks")
        
        # Verification step to ensure we haven't lost content
        original_content_length = sum(len(chunk) for chunk in chunks)
        final_content_length = sum(len(chunk) for chunk in final_chunks)
        
        # Account for added paragraph separators during merging
        separator_count = sum(chunk.count(self.paragraph_separator) for chunk in final_chunks) - sum(chunk.count(self.paragraph_separator) for chunk in chunks)
        expected_length = original_content_length + separator_count * len(self.paragraph_separator)
        
        if final_content_length < original_content_length:
            self._log(f"WARNING: Content loss detected! Original: {original_content_length}, Final: {final_content_length}")
            # Fall back to original chunks if we lost content
            return chunks
            
        return final_chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text into semantically and structurally meaningful chunks, ensuring no content is lost."""
        if not text:
            return []
            
        original_text = text.strip()
        if not original_text:
            return []
            
        self._log(f"Splitting text of length {len(text)}")
        
        # First, try enhanced splitting with structural awareness
        try:
            chunks = self._segment_with_structure(text)
            if self._verify_content_preservation(original_text, chunks):
                return chunks
            else:
                self._log("Content preservation check failed for structure-based splitting")
        except Exception as e:
            self._log(f"Enhanced chunking failed: {str(e)}")
            self._log("Falling back to basic chunking")
        
        # Fallback to basic paragraph splitting
        try:
            chunks = self._segment_text_basic(text)
            if self._verify_content_preservation(original_text, chunks):
                return chunks
            else:
                self._log("Content preservation check failed for basic splitting")
        except Exception as e:
            self._log(f"Basic chunking also failed: {str(e)}")
        
        # Ultimate fallback: just return the original text as a single chunk
        # This ensures we never lose content even if all chunking methods fail
        self._log("Using emergency fallback: returning text as a single chunk")
        return [original_text]
        
    def _verify_content_preservation(self, original_text: str, chunks: List[str]) -> bool:
        """
        Verify that all meaningful content from the original text is preserved in the chunks.
        
        This is an approximate check that looks for significant content loss.
        """
        if not chunks:
            return False
            
        # Remove excess whitespace for comparison
        original_normalized = re.sub(r'\s+', ' ', original_text).strip()
        
        # Join chunks with a separator for comparison
        chunks_normalized = ' '.join([re.sub(r'\s+', ' ', chunk).strip() for chunk in chunks])
        
        # Check if the normalized content length is reasonably preserved
        # Allow for some minor differences due to whitespace normalization
        original_length = len(original_normalized)
        chunks_length = len(chunks_normalized)
        
        # Calculate content preservation ratio (should be close to 1.0)
        # We use character count as an approximation
        ratio = chunks_length / original_length if original_length > 0 else 0
        
        # Log the content preservation statistics
        self._log(f"Content preservation check: Original: {original_length} chars, Chunks: {chunks_length} chars, Ratio: {ratio:.2f}")
        
        # We consider it successful if we preserved at least 90% of the content
        # This threshold can be adjusted based on your specific needs
        return ratio >= 0.9
        
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                # Add structure info to metadata
                if self.respect_structure:
                    structure_info = []
                    for pattern_name, pattern in self.patterns.items():
                        if pattern.search(chunk):
                            structure_info.append(pattern_name)
                    
                    if structure_info:
                        doc_metadata["structure_elements"] = structure_info
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantically and structurally meaningful chunks."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.create_documents(texts, metadatas)
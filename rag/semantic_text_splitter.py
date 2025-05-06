"""
Enhanced Semantic Text Splitter optimized for narrative texts like Harry Potter books

This module provides a semantic text splitter that divides documents
based on narrative structure, dialogue, scene breaks, and semantic cohesion
rather than arbitrary character counts.
"""

import nltk, torch, re
from typing import List, Optional, Dict
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
    Split text based on narrative structure and semantic meaning for fiction texts.
    Optimized for Harry Potter books with scene detection, dialogue preservation,
    and natural narrative breaks.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",  # Upgraded model
        similarity_threshold: float = 0.6,  # Higher threshold for narrative consistency
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,  # Larger chunk size for narrative context
        chunk_overlap: int = 100,  # Increased overlap for continuity
        paragraph_separator: str = "\n\n",
        sentence_separator: str = "\n",
        respect_structure: bool = True,
        preserve_dialogue: bool = True,  # New parameter for dialogue handling
        verbose: bool = False
    ):
        """
        Initialize the semantic text splitter optimized for narrative texts.
        
        Args:
            embedding_model: The embedding model to use for semantic similarity
            similarity_threshold: Threshold for determining semantic similarity (0-1)
            min_chunk_size: Minimum size of chunks in characters
            max_chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            paragraph_separator: String that separates paragraphs
            sentence_separator: String that separates sentences within paragraphs
            respect_structure: Whether to respect document structure
            preserve_dialogue: Whether to keep dialogue exchanges together
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
        self.preserve_dialogue = preserve_dialogue
        self.verbose = verbose
        
        # Structural patterns optimized for narrative fiction
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
        """Compile regex patterns optimized for narrative fiction texts."""
        return {
            # Chapter markers (common in Harry Potter books)
            'chapter_heading': re.compile(r'^(?:CHAPTER|Chapter)\s+[A-Z0-9]+(?:\s+[A-Z].*)?$', re.MULTILINE),
            
            # Scene breaks (common in novels)
            'scene_break': re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE),
            
            # Dialogue markers
            'dialogue_start': re.compile(r'^\s*[\'"].*', re.MULTILINE),
            'dialogue_continuation': re.compile(r'.*[,:][\s]*[\'"]\s*\w+.*$', re.MULTILINE),
            
            # Character indicators (common in Harry Potter)
            'character_indicator': re.compile(r'\b(Harry|Ron|Hermione|Dumbledore|Snape|Hagrid|Voldemort)\b\s*(?:said|asked|replied|shouted|whispered|muttered|exclaimed)', re.IGNORECASE),
            
            # Location changes
            'location_change': re.compile(r'\b(?:meanwhile|elsewhere|later|back at|in the|at the)\b.*', re.IGNORECASE),
            
            # Time transitions
            'time_transition': re.compile(r'\b(?:the next|that|the following|the previous|earlier|later)\s+(?:day|morning|afternoon|evening|night|week|month|year)\b', re.IGNORECASE),
            
            # PDF page breaks and headers (for handling scanned books)
            'page_break': re.compile(r'\f'),
            'header': re.compile(r'^[^.!?]*(?:[Pp]age|[Cc]hapter)\s*\d+.*\n'),
            'footer': re.compile(r'\n[^.!?]*(?:[Pp]age|[Cc]hapter)\s*\d+.*$'),
        }
    
    def _log(self, message: str) -> None:
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticTextSplitter] {message}")
    
    def _is_structural_boundary(self, text: str) -> bool:
        """Determine if the text contains a narrative structural boundary."""
        if not self.respect_structure:
            return False
            
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(text):
                self._log(f"Found structural boundary: {pattern_name}")
                return True
        return False
    
    def _is_dialogue_unit(self, text: str) -> bool:
        """Check if this text represents a dialogue exchange that should be preserved."""
        if not self.preserve_dialogue:
            return False
        
        # Count quotation marks - uneven counts suggest continuation
        quote_count = text.count('"') + text.count("'")
        if quote_count % 2 != 0:
            return True
            
        # Check for dialogue patterns
        if self.patterns['dialogue_start'].search(text) and self.patterns['dialogue_continuation'].search(text):
            return True
            
        # Check for character speech indicators
        if self.patterns['character_indicator'].search(text):
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
        """Determine if two text segments are semantically similar in narrative context."""
        # For very short texts, consider them similar to avoid over-chunking
        if len(text1) < 70 or len(text2) < 70:
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
    
    def _segment_narrative_text(self, text: str) -> List[str]:
        """
        Segment narrative text into chunks based on narrative structure, dialogue, and semantic cohesion.
        Optimized for fiction like Harry Potter books.
        """
        if not text.strip():
            return []
        
        # Split text by paragraphs first
        paragraphs = text.split(self.paragraph_separator)
        
        # Keep track of original paragraph content
        original_paragraphs = [p for p in paragraphs]
        
        # Filter out empty paragraphs for processing
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [text] if text.strip() else []
        
        # Process paragraphs with narrative structure awareness
        chunks = []
        current_chunk = ""
        dialogue_context = ""  # Track ongoing dialogue
        chapter_start = False  # Track if we're at the start of a chapter
        
        for i, para in enumerate(paragraphs):
            # Check if this paragraph starts a new chapter
            if self.patterns['chapter_heading'].search(para):
                # Always start a new chunk for chapters
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                chapter_start = True
                dialogue_context = ""
                continue
                
            # Check if this is a scene break
            if self.patterns['scene_break'].search(para):
                # End the current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                dialogue_context = ""
                continue
            
            # Check for ongoing dialogue
            is_dialogue = self._is_dialogue_unit(para)
            
            # Special handling for dialogue
            if is_dialogue:
                # If we have ongoing dialogue, check if this continues it
                if dialogue_context and self._is_semantically_similar(dialogue_context, para):
                    # Continue the dialogue in current chunk if it fits
                    if len(current_chunk) + len(para) + len(self.paragraph_separator) <= self.max_chunk_size:
                        if current_chunk:
                            current_chunk += self.paragraph_separator + para
                        else:
                            current_chunk = para
                        dialogue_context += " " + para
                        continue
                
                # Start tracking a new dialogue exchange
                dialogue_context = para
            
            # Regular paragraph handling with enhanced narrative continuity
            
            # Check size constraints
            if len(current_chunk) + len(para) + len(self.paragraph_separator) > self.max_chunk_size:
                # Current chunk would be too large - save it and start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            
            # Check semantic similarity for narrative continuity
            if current_chunk and not chapter_start and not self._is_semantically_similar(current_chunk, para):
                chunks.append(current_chunk)
                current_chunk = ""
                dialogue_context = ""
            
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += self.paragraph_separator + para
            else:
                current_chunk = para
                
            # Reset chapter_start flag
            chapter_start = False
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # If we have no chunks (unlikely), return the original text
        if not chunks:
            self._log("Warning: No chunks created during narrative-based splitting")
            return [text]
        
        # Ensure chunks meet minimum size by merging small chunks
        merged_chunks = self._merge_small_chunks(chunks)
        
        # Verify content preservation
        original_content = ''.join(original_paragraphs).strip()
        merged_content = ''.join(merged_chunks).strip()
        
        # If we lost significant content, fall back to the pre-merged chunks
        if len(merged_content) < len(original_content) * 0.9:
            self._log(f"Warning: Content loss detected during narrative chunking")
            if len(''.join(chunks).strip()) >= len(original_content) * 0.9:
                return chunks
            else:
                # Ultimate fallback if even pre-merged chunks lost content
                return [text]
        
        return merged_chunks
    
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
        
        self._log(f"Created {len(final_chunks)} narrative chunks after merging")
        
        return final_chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text into narrative-aware chunks optimized for fiction like Harry Potter."""
        if not text:
            return []
            
        original_text = text.strip()
        if not original_text:
            return []
            
        # Try narrative chunking optimized for fiction
        try:
            chunks = self._segment_narrative_text(text)
            if self._verify_content_preservation(original_text, chunks):
                return chunks
            else:
                self._log("Content preservation check failed for narrative splitting")
        except Exception as e:
            self._log(f"Narrative chunking failed: {str(e)}")
        
        # Ultimate fallback: just return the original text as a single chunk
        self._log("Using emergency fallback: returning text as a single chunk")
        return [original_text]
        
    def _verify_content_preservation(self, original_text: str, chunks: List[str]) -> bool:
        """Verify that all meaningful content from the original text is preserved in the chunks."""
        if not chunks:
            return False
            
        # Remove excess whitespace for comparison
        original_normalized = re.sub(r'\s+', ' ', original_text).strip()
        
        # Join chunks with a separator for comparison
        chunks_normalized = ' '.join([re.sub(r'\s+', ' ', chunk).strip() for chunk in chunks])
        
        # Calculate content preservation ratio (should be close to 1.0)
        original_length = len(original_normalized)
        chunks_length = len(chunks_normalized)
        ratio = chunks_length / original_length if original_length > 0 else 0
        
        # We consider it successful if we preserved at least 90% of the content
        return ratio >= 0.9
        
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts with enhanced metadata for narrative context."""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                # Add narrative structure info to metadata
                if self.respect_structure:
                    # Detect chapter headings
                    if self.patterns['chapter_heading'].search(chunk):
                        chapter_match = self.patterns['chapter_heading'].search(chunk)
                        doc_metadata["is_chapter_start"] = True
                        doc_metadata["chapter_heading"] = chapter_match.group(0) if chapter_match else "Unknown Chapter"
                    
                    # Detect scene transitions
                    if self.patterns['scene_break'].search(chunk):
                        doc_metadata["contains_scene_break"] = True
                    
                    # Detect if chunk contains dialogue
                    if '"' in chunk or "'" in chunk:
                        doc_metadata["contains_dialogue"] = True
                    
                    # Check for character mentions
                    character_matches = re.findall(r'\b(Harry|Ron|Hermione|Dumbledore|Snape|Hagrid|Voldemort)\b', chunk, re.IGNORECASE)
                    if character_matches:
                        doc_metadata["characters_mentioned"] = list(set([match.title() for match in character_matches]))
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into narrative-aware chunks with enhanced metadata."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.create_documents(texts, metadatas)
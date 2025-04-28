"""
Semantic Text Splitter for RAGbot

This module provides a semantic text splitter that divides documents
based on meaning rather than arbitrary character counts.
"""

import re
import nltk
import torch
from typing import List, Dict, Any, Optional, Callable
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Make sure all necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)

class SemanticTextSplitter(TextSplitter):
    """
    Split text based on semantic meaning rather than fixed-size chunks.
    
    This splitter uses embeddings to detect semantic shifts in content and
    creates more natural chunks based on content meaning.
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
        verbose: bool = False
    ):
        """
        Initialize the semantic text splitter.
        
        Args:
            embedding_model (str): Model to use for embeddings
            similarity_threshold (float): Threshold to determine semantic similarity
            min_chunk_size (int): Minimum size of chunks in characters
            max_chunk_size (int): Maximum size of chunks in characters
            chunk_overlap (int): Number of characters to overlap between chunks
            paragraph_separator (str): String that separates paragraphs
            sentence_separator (str): String that separates sentences
            verbose (bool): Whether to print verbose output
        """
        # Initialize the parent class with required parameters
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
        self.verbose = verbose
        
        # Initialize the embedding model with proper device detection
        self._log(f"Initializing embedding model: {embedding_model}")
        try:
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"Using device: {device}")
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        except Exception as e:
            self._log(f"Error initializing model with device detection: {str(e)}")
            # Fall back to basic initialization
            self._log("Falling back to basic model initialization")
            self.embedding_model = SentenceTransformer(embedding_model)
    
    def _log(self, message: str) -> None:
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticTextSplitter] {message}")
            
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Handle zero vectors
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            return 0.0
        
        try:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)  # Ensure we return a Python float
        except Exception as e:
            self._log(f"Error calculating similarity: {str(e)}")
            return 0.0  # Return 0 similarity on error
    
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
            
            return similarity >= self.similarity_threshold
        except Exception as e:
            self._log(f"Error in semantic similarity check: {str(e)}")
            # On error, default to considering them similar to avoid chunking errors
            return True
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text (str): The text to split
            
        Returns:
            List[str]: A list of text chunks
        """
        if not text:
            return []
            
        self._log(f"Splitting text of length {len(text)}")
        
        # First, try splitting by semantic meaning
        try:
            chunks = self._segment_text(text)
            if chunks:
                return chunks
        except Exception as e:
            self._log(f"Semantic chunking failed: {str(e)}")
            self._log("Falling back to basic chunking")
        
        # Fallback to simple paragraph splitting if semantic chunking fails
        paragraphs = text.split(self.paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [text] if text.strip() else []
        
        # Combine paragraphs to meet minimum size when possible
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            
            if current_chunk:
                current_chunk += self.paragraph_separator + paragraph
            else:
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks or [text]
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into meaningful chunks based on semantic similarity.
        
        This method divides text first by paragraphs, then by sentences,
        and combines them based on semantic similarity until reaching
        the maximum chunk size.
        """
        # Split by paragraphs first
        paragraphs = text.split(self.paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If we have no paragraphs, return the original text
        if not paragraphs:
            return [text] if text.strip() else []
            
        chunks = []
        current_chunk = ""
        last_added_text = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split it into sentences
            if len(paragraph) > self.max_chunk_size:
                try:
                    sentences = nltk.sent_tokenize(paragraph)
                except Exception as e:
                    self._log(f"Error in sentence tokenization: {str(e)}")
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
                            if self.chunk_overlap > 0 and last_added_text:
                                current_chunk = last_added_text
                            else:
                                current_chunk = ""
                    
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
                # For shorter paragraphs, add them as is if semantically similar
                if len(current_chunk) + len(paragraph) + 1 > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Keep overlap for context
                        if self.chunk_overlap > 0 and last_added_text:
                            current_chunk = last_added_text
                        else:
                            current_chunk = ""
                
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
        
        # Make sure all chunks meet minimum size where possible
        final_chunks = []
        small_chunk = ""
        
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                if small_chunk:
                    small_chunk += self.paragraph_separator + chunk
                else:
                    small_chunk = chunk
                
                if len(small_chunk) >= self.min_chunk_size:
                    final_chunks.append(small_chunk)
                    small_chunk = ""
            else:
                final_chunks.append(chunk)
        
        # Add any remaining small chunk
        if small_chunk:
            if final_chunks:
                final_chunks[-1] += self.paragraph_separator + small_chunk
            else:
                final_chunks.append(small_chunk)
        
        self._log(f"Split text into {len(final_chunks)} semantic chunks")
        return final_chunks
        
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Create documents from a list of texts.
        
        Args:
            texts (List[str]): List of texts to split and create documents from
            metadatas (Optional[List[dict]]): Optional metadata for each text
            
        Returns:
            List[Document]: A list of documents
        """
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically meaningful chunks.
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: A list of split documents
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.create_documents(texts, metadatas)
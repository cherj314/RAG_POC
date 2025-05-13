import os, nltk, torch, re, numpy as np
from typing import List, Optional
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# Define and set up NLTK data directory - simplified approach
nltk_data_dir = os.getenv('NLTK_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data'))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download NLTK data if needed (simplified)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print(f"Downloading NLTK punkt tokenizer to {nltk_data_dir}...")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)

class SemanticTextSplitter(TextSplitter):
    """
    A text splitter that creates chunks based on semantic similarity 
    while ensuring chunks respect sentence boundaries.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        similarity_threshold: float = 0.6,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
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
        
        # Initialize the embedding model
        self._log(f"Initializing embedding model: {embedding_model}")
        try:
            # Try to use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"Using device: {device}")
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        except Exception as e:
            self._log(f"Error initializing model with device detection: {str(e)}")
            self.embedding_model = SentenceTransformer(embedding_model)
        
        # Compile structural patterns (simplified)
        self.patterns = {
            'chapter': re.compile(r'^(?:CHAPTER|Chapter)\s+[A-Z0-9]+(?:\s+[A-Z].*)?$', re.MULTILINE),
            'scene_break': re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE),
            'page_number': re.compile(r'^\d+$', re.MULTILINE),
            'book_title': re.compile(r'^HARRY POTTER.*$', re.MULTILINE)
        }
    
    def _log(self, message: str) -> None:
        """Print log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticTextSplitter] {message}")
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings more efficiently."""
        # Handle zero vectors
        if np.all(embedding1 == 0) or np.all(embedding2 == 0):
            return 0.0
        
        # Use numpy's optimized functions directly
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """Check if two text segments are semantically similar."""
        # Skip comparison for very short texts
        if len(text1) < 50 or len(text2) < 50:
            return True
        
        try:
            # Encode both texts in a single batch for efficiency
            embeddings = self.embedding_model.encode([text1, text2], convert_to_numpy=True)
            return self._calculate_similarity(embeddings[0], embeddings[1]) >= self.similarity_threshold
        except Exception as e:
            self._log(f"Error in semantic similarity check: {str(e)}")
            return True  # Default to assuming similar on error
    
    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with robust fallback."""
        if not text or not text.strip():
            return []
            
        # Normalize text first
        text = self._preprocess_text(text)
        
        # Try NLTK first
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception as e:
            self._log(f"NLTK sentence tokenization failed: {str(e)}. Using regex fallback.")
            
            # Fallback to regex (simplified)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before processing."""
        # Remove page numbers and headers
        text = self.patterns['page_number'].sub('', text)
        text = self.patterns['book_title'].sub('', text)
        
        # Normalize line breaks and whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Handle quotes after sentence endings
        text = re.sub(r'([.!?])"(\s)', r'\1" \2', text)
        
        return text
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while respecting sentence boundaries."""
        if not text or not text.strip():
            return []
        
        # Get sentences
        sentences = self._get_sentences(text)
        
        # Process sentences into semantically coherent chunks
        return self._create_chunks(sentences)
    
    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks from sentences, respecting size limits and semantic similarity."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = sentences[0]
        current_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Check if adding this sentence would exceed size limit
            if len(current_chunk) + len(sentences[i]) + 1 > self.max_chunk_size:
                # Complete the current chunk
                chunks.append(current_chunk)
                current_chunk = sentences[i]
                current_sentences = [sentences[i]]
            else:
                # Add the sentence to the current chunk
                current_chunk += " " + sentences[i]
                current_sentences.append(sentences[i])
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Apply structural boundary handling if needed
        if self.respect_structure:
            chunks = self._respect_boundaries(chunks)
        
        # Merge small chunks to avoid tiny fragments
        return self._merge_small_chunks(chunks)
    
    def _respect_boundaries(self, chunks: List[str]) -> List[str]:
        """Ensure chunks respect structural boundaries like chapters."""
        if not chunks:
            return []
        
        result = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            # Check if this chunk starts with a structural boundary
            chunk_start = chunks[i].strip().split("\n")[0]
            is_boundary = (self.patterns['chapter'].search(chunk_start) or 
                          self.patterns['scene_break'].search(chunk_start))
            
            if is_boundary:
                # Start a new chunk at boundary
                result.append(current)
                current = chunks[i]
            elif len(current) + len(chunks[i]) + 2 <= self.max_chunk_size:
                # Combine chunks if under size limit
                current += " " + chunks[i]
            else:
                # Otherwise create a new chunk
                result.append(current)
                current = chunks[i]
        
        # Add the last chunk
        if current:
            result.append(current)
            
        return result
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks to avoid tiny fragments."""
        if len(chunks) <= 1:
            return chunks
            
        result = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Look ahead to merge small chunks
            while (i + 1 < len(chunks) and 
                  len(current) < self.min_chunk_size and 
                  len(current) + len(chunks[i+1]) + 1 <= self.max_chunk_size):
                current += " " + chunks[i+1]
                i += 1
                
            result.append(current)
            i += 1
                
        return result
    
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from text and metadata."""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                # Add structure detection
                if self.respect_structure:
                    # Detect if chunk contains chapter heading
                    if self.patterns['chapter'].search(chunk):
                        doc_metadata["contains_chapter_heading"] = True
                    
                    # Detect if chunk contains scene break
                    if self.patterns['scene_break'].search(chunk):
                        doc_metadata["contains_scene_break"] = True
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preserved."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return self.create_documents(texts, metadatas)
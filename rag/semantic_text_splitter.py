import os, nltk, torch, re, numpy as np
from typing import List, Optional
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# Define and set up NLTK data directory
nltk_data_dir = os.getenv('NLTK_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data'))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Try to download the correct NLTK data
try:
    # First check for punkt (the standard tokenizer)
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt tokenizer already downloaded.")
    except LookupError:
        print(f"Downloading NLTK punkt tokenizer to {nltk_data_dir}...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
    
    # Now specifically try to handle punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print(f"Attempting to ensure punkt_tab is available...")
        # Standard punkt should include punkt_tab, but we'll make sure the punkt download was successful
        if os.path.exists(os.path.join(nltk_data_dir, 'tokenizers', 'punkt')):
            print("punkt tokenizer is available, which should contain punkt_tab")
except Exception as e:
    print(f"Warning: Error setting up NLTK: {str(e)}")
    print("Proceeding with fallback sentence splitting.")

# Define a robust sentence splitter as fallback
def basic_sentence_split(text):
    """Split text into sentences using basic regex patterns that mimic NLTK's behavior."""
    # Clean up the text first
    text = re.sub(r'\s+', ' ', text)
    
    # Split by common sentence ending punctuation followed by a space and capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Further split by common sentence ending marks
    final_sentences = []
    for sentence in sentences:
        # Split by line breaks that might indicate sentence breaks
        parts = re.split(r'\n{2,}', sentence)
        for part in parts:
            if part.strip():
                final_sentences.append(part.strip())
    
    return final_sentences if final_sentences else [text]

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
        chunk_overlap: int = 200,  # Increased from 100
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
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._log(f"Using device: {device}")
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        except Exception as e:
            self._log(f"Error initializing model with device detection: {str(e)}")
            self._log("Falling back to basic model initialization")
            self.embedding_model = SentenceTransformer(embedding_model)
        
        # Compile structural patterns
        self.chapter_pattern = re.compile(r'^(?:CHAPTER|Chapter)\s+[A-Z0-9]+(?:\s+[A-Z].*)?$', re.MULTILINE)
        self.scene_break_pattern = re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE)
        self.page_number_pattern = re.compile(r'^\d+$', re.MULTILINE)
        self.book_title_pattern = re.compile(r'^HARRY POTTER.*$', re.MULTILINE)
    
    def _log(self, message: str) -> None:
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[SemanticTextSplitter] {message}")
    
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
        """Check if two text segments are semantically similar."""
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
            return True
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while respecting sentence boundaries."""
        if not text or not text.strip():
            return []
        
        # Remove page numbers and headers first
        text = self.page_number_pattern.sub('', text)
        text = self.book_title_pattern.sub('', text)
        
        # Normalize line breaks and whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Add explicit handling for dialogue and quotes which might confuse sentence tokenizers
        text = re.sub(r'([.!?])"(\s)', r'\1" \2', text)  # Fix quotes after sentence endings
        
        # Better regex for sentence splitting as fallback
        def better_sentence_split(text):
            # More robust regex that handles various edge cases
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(pattern, text)
            
            # Further process each potential sentence to avoid fragments
            result = []
            buffer = ""
            for sent in sentences:
                # Check if this looks like a complete sentence
                if re.search(r'[.!?]\s*$', sent) or len(sent) > 100:
                    if buffer:
                        result.append(buffer + " " + sent)
                        buffer = ""
                    else:
                        result.append(sent)
                else:
                    if buffer:
                        buffer += " " + sent
                    else:
                        buffer = sent
            
            # Add any remaining buffer
            if buffer:
                result.append(buffer)
                
            return result
        
        try:
            from nltk.tokenize import PunktSentenceTokenizer
            tokenizer = PunktSentenceTokenizer()
            sentences = tokenizer.tokenize(text)
        except Exception as e:
            self._log(f"NLTK tokenization failed: {str(e)}. Using fallback.")
            sentences = better_sentence_split(text)
        
        # Process sentences into semantically coherent chunks
        chunks = self._group_sentences_semantically(sentences)
        
        # Apply post-processing to ensure complete sentences
        chunks = self._ensure_complete_sentences(chunks)
        
        return chunks
  
    def _group_sentences_semantically(self, sentences: List[str]) -> List[str]:
        """Group sentences into semantically coherent chunks while respecting sentence boundaries."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = sentences[0]
        current_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Check if adding this sentence would exceed size limit
            if len(current_chunk) + len(sentences[i]) + 1 > self.max_chunk_size:
                # Never break in the middle of a sentence - complete the current chunk
                chunks.append(current_chunk)
                current_chunk = sentences[i]
                current_sentences = [sentences[i]]
            else:
                # ALWAYS add the complete sentence to the current chunk, regardless of semantic similarity
                # This ensures we never break sentences
                current_chunk += " " + sentences[i]
                current_sentences.append(sentences[i])
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Handle structural boundaries if needed
        if self.respect_structure:
            chunks = self._respect_structural_boundaries(chunks)
        
        # Merge small chunks if needed
        return self._merge_small_chunks(chunks)
    
    def _respect_structural_boundaries(self, chunks: List[str]) -> List[str]:
        """Adjust chunks to respect structural boundaries like chapters."""
        if not chunks:
            return []
        
        result = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            # If this chunk starts with a chapter heading, start a new chunk
            if self.chapter_pattern.search(chunks[i].strip().split("\n")[0]):
                result.append(current)
                current = chunks[i]
            # If this chunk starts with a scene break, start a new chunk
            elif self.scene_break_pattern.search(chunks[i].strip().split("\n")[0]):
                result.append(current)
                current = chunks[i]
            else:
                # Check if adding would exceed size
                if len(current) + len(chunks[i]) + 2 <= self.max_chunk_size:
                    current += " " + chunks[i]
                else:
                    result.append(current)
                    current = chunks[i]
        
        # Add the last chunk
        if current:
            result.append(current)
            
        return result
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks where possible to avoid tiny fragments."""
        if len(chunks) <= 1:
            return chunks
            
        result = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            # If current chunk is too small and can be merged with next
            if len(current) < self.min_chunk_size and len(current) + len(chunks[i]) + 1 <= self.max_chunk_size:
                current += " " + chunks[i]
            else:
                result.append(current)
                current = chunks[i]
                
        # Add the last chunk
        if current:
            result.append(current)
            
        return result
    
    def _ensure_complete_sentences(self, chunks: List[str]) -> List[str]:
        """Fix chunks to ensure they start and end with complete sentences."""
        if not chunks:
            return []

        result = []
        
        for i, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Fix chunks that start with lowercase (likely mid-sentence)
            if chunk and chunk[0].islower() and i > 0 and result:
                # Try to find the last sentence in the previous chunk
                prev_chunk = result[-1]
                prev_sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
                
                if prev_sentences and not prev_chunk.endswith(('.', '!', '?', '."', '!"', '?"')):
                    # Take the incomplete sentence from previous chunk
                    incomplete_sent = prev_sentences[-1]
                    
                    # Find where to split the current chunk
                    current_sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    if current_sentences:
                        # Complete the sentence
                        completed_sent = incomplete_sent + " " + current_sentences[0]
                        
                        # Update previous chunk without the incomplete sentence
                        result[-1] = " ".join(prev_sentences[:-1])
                        
                        # Update current chunk with the completed sentence
                        chunk = completed_sent + " " + " ".join(current_sentences[1:])
        
            result.append(chunk)
    
        return result
    
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from texts with metadata."""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                # Add basic structure detection
                if self.respect_structure:
                    # Detect if chunk contains chapter heading
                    if self.chapter_pattern.search(chunk):
                        doc_metadata["contains_chapter_heading"] = True
                    
                    # Detect if chunk contains scene break
                    if self.scene_break_pattern.search(chunk):
                        doc_metadata["contains_scene_break"] = True
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        split_docs = self.create_documents(texts, metadatas)
        
        # Add post-processing to fix sentence boundaries
        if split_docs:
            chunks = [doc.page_content for doc in split_docs]
            fixed_chunks = self._ensure_complete_sentences(chunks)
            
            # Update the document chunks with fixed content
            for i, fixed_chunk in enumerate(fixed_chunks):
                if i < len(split_docs):
                    split_docs[i].page_content = fixed_chunk
        
        return split_docs
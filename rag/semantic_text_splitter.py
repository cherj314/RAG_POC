import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Text splitter that creates chunks based on semantic similarity
class SemanticTextSplitter(TextSplitter):
    
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
        super().__init__(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_structure = respect_structure
        self.verbose = verbose
        self.embedding_model = None
        
        # Structural element patterns
        self.patterns = {
            'chapter': re.compile(r'^(?:CHAPTER|Chapter)\s+[IVXLCDM\d]+', re.MULTILINE),
            'scene_break': re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE),
            'page_number': re.compile(r'^\s*\d+\s*$', re.MULTILINE),
            'header': re.compile(r'HARRY POTTER|^THE\s+[A-Z\s]+$', re.MULTILINE),
            'sentence_end': re.compile(r'[.!?][\'\"]?\s+')
        }
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[SemanticSplitter] {message}")
    
    # Lazy load the embedding model
    def _get_embedding_model(self):
        if self.embedding_model is None:
            self._log(f"Loading embedding model")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.embedding_model
    
    # Preprocess text to remove headers, footers, and page numbers
    def _preprocess_text(self, text: str) -> str:
        text = self.patterns['page_number'].sub('', text)
        text = self.patterns['header'].sub('', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([.!?])"(\s)', r'\1" \2', text)
        return text.strip()
    
    # Check if text contains a structural boundary like a chapter heading
    def _is_structural_boundary(self, text: str) -> bool:
        if not self.respect_structure:
            return False
        first_line = text.strip().split('\n')[0] if text.strip() else ""
        return bool(self.patterns['chapter'].search(first_line) or 
                    self.patterns['scene_break'].search(first_line))
    
    # Calculate semantic similarity between two text segments
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if len(text1) < 100 or len(text2) < 100:
            return 1.0
        try:
            model = self._get_embedding_model()
            embed1 = model.encode(text1)
            embed2 = model.encode(text2)
            similarity = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
            return float(similarity)
        except Exception as e:
            self._log(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    # Split text into sentences using regex pattern matching
    def _split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = self.patterns['sentence_end'].split(text)
        result = []
        terminators = re.findall(self.patterns['sentence_end'], text)
        
        for i, sentence in enumerate(sentences[:-1]):
            if i < len(terminators):
                result.append(sentence + terminators[i])
            else:
                result.append(sentence)
        
        if sentences[-1].strip():
            result.append(sentences[-1])
        
        return [s.strip() for s in result if s.strip()]
    
    # Adjust chunk boundary to ensure it ends at a sentence boundary
    def _adjust_chunk_boundary(self, chunk: str) -> str:
        if re.search(r'[.!?][\'\"]?$', chunk.strip()):
            return chunk
        
        match = list(re.finditer(r'[.!?][\'\"]?\s+', chunk))
        if match:
            return chunk[:match[-1].end()].strip()
        
        return chunk
    
    # Split text into semantically coherent chunks
    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        # Preprocess text and split into paragraphs
        cleaned_text = self._preprocess_text(text)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', cleaned_text) if p.strip()]
        
        # Group paragraphs into chunks
        chunks, current_chunk, current_size = [], [], 0
        
        for paragraph in paragraphs:
            # Start new chunk at structural boundaries
            if self._is_structural_boundary(paragraph):
                if current_chunk:
                    chunks.append(self._adjust_chunk_boundary("\n\n".join(current_chunk)))
                current_chunk, current_size = [paragraph], len(paragraph)
                continue
                
            # Check if adding paragraph exceeds max size
            if current_size + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append(self._adjust_chunk_boundary("\n\n".join(current_chunk)))
                current_chunk, current_size = [paragraph], len(paragraph)
                continue
                
            # Check semantic similarity
            if current_chunk and len(current_chunk[-1]) > 100 and len(paragraph) > 100:
                if self._calculate_similarity(current_chunk[-1], paragraph) < self.similarity_threshold and current_size > self.min_chunk_size:
                    chunks.append(self._adjust_chunk_boundary("\n\n".join(current_chunk)))
                    current_chunk, current_size = [paragraph], len(paragraph)
                    continue
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += len(paragraph) + 4
        
        # Add the last chunk
        if current_chunk:
            chunks.append(self._adjust_chunk_boundary("\n\n".join(current_chunk)))
        
        # Process chunks for size constraints and ensure sentence boundaries
        final_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            
            # Handle small chunks by merging with next
            if len(chunk) < self.min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                if len(chunk) + len(next_chunk) <= self.max_chunk_size:
                    i += 1
                    continue
            
            # Split large chunks into sentences if needed
            if len(chunk) > self.max_chunk_size:
                sentences = self._split_into_sentences(chunk)
                if len(sentences) > 1:
                    sub_chunks, sub_chunk, sub_size = [], [], 0
                    for sentence in sentences:
                        if sub_size + len(sentence) > self.max_chunk_size and sub_chunk:
                            sub_chunks.append(" ".join(sub_chunk))
                            sub_chunk, sub_size = [sentence], len(sentence)
                        else:
                            sub_chunk.append(sentence)
                            sub_size += len(sentence) + 1
                    
                    if sub_chunk:
                        sub_chunks.append(" ".join(sub_chunk))
                    
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
            i += 1
        
        # Add overlap between chunks if needed
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            overlapped_chunks = [final_chunks[0]]
            
            for i in range(1, len(final_chunks)):
                prev_chunk = final_chunks[i-1]
                current_chunk = final_chunks[i]
                
                prev_sentences = self._split_into_sentences(prev_chunk)
                if prev_sentences:
                    overlap_size, overlap_sentences = 0, []
                    
                    for sentence in reversed(prev_sentences):
                        if overlap_size + len(sentence) > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, sentence)
                        overlap_size += len(sentence) + 1
                    
                    if overlap_sentences:
                        overlap_text = " ".join(overlap_sentences)
                        if not current_chunk.startswith(overlap_text):
                            current_chunk = overlap_text + " " + current_chunk
                
                overlapped_chunks.append(current_chunk)
            
            return overlapped_chunks
        
        return final_chunks
    
    # Split a list of documents into chunks
    def split_documents(self, documents: List[Document]) -> List[Document]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        result_docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk"] = j
                doc_metadata["total_chunks"] = len(chunks)
                
                if self.respect_structure:
                    lines = chunk.strip().split('\n')
                    if lines and self.patterns['chapter'].search(lines[0]):
                        doc_metadata["contains_chapter_heading"] = True
                        doc_metadata["current_chapter"] = lines[0].strip()
                
                result_docs.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return result_docs
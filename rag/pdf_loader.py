import os, re
import fitz   # type: ignore
from typing import List, Dict, Any
from langchain.docstore.document import Document

class PDFLoader:
    def __init__(self, file_path: str, encoding: str = "utf-8", verbose: bool = False, 
                 max_workers: int = 4, batch_size: int = 10, extract_images: bool = False):
        self.file_path = file_path
        self.encoding = encoding
        self.verbose = verbose
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.extract_images = extract_images
        self._current_chapter = None
        
        # Patterns for detecting elements to exclude
        self.patterns = {
            # Match various forms of Harry Potter style chapter headings
            'chapter_heading': re.compile(r'(?:—\s*)?CHAPTER\s+(?:[A-Z]+|\d+)(?:\s+[A-Z]+)?(?:\s*—)?', re.IGNORECASE),
            'page_number': re.compile(r'^\s*\d+\s*$'),
            # Pattern to identify sentence boundaries
            'sentence_end': re.compile(r'[.!?][\'\"]?\s+')
        }
    
    def _log(self, message: str) -> None:
        """Print log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[PDFLoader] {message}")
    
    def _is_chapter_heading(self, text: str) -> bool:
        """Check if text is a chapter heading."""
        text = text.strip()
        return bool(self.patterns['chapter_heading'].match(text))
    
    def _is_header(self, text: str, y_pos: float, page_height: float) -> bool:
        """Determine if text is a header based on position."""
        text = text.strip()
        
        # Check position - if very close to top of page (within 6%)
        if y_pos < page_height * 0.06:
            return True
        
        return False
    
    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text starts and ends with complete sentences."""
        if not text.strip():
            return text
            
        # Check if text ends with a sentence terminator
        if not re.search(r'[.!?][\'\"]?$', text.strip()):
            # Find the last sentence boundary
            match = list(re.finditer(r'[.!?][\'\"]?\s+', text))
            if match:
                # Get the position of the last sentence terminator
                last_terminator_pos = match[-1].end()
                # Return the text up to that position
                text = text[:last_terminator_pos].strip()
        
        # Check if text starts with a capital letter
        # If not, it might be in the middle of a sentence
        if text and not text[0].isupper() and not text[0].isdigit():
            # Try to find the next sentence start
            match = re.search(r'[.!?][\'\"]?\s+([A-Z0-9])', text)
            if match:
                # Start from the beginning of the next sentence
                text = text[match.start(1)-1:].strip()
        
        return text
    
    def _detect_chapter_structure(self, page: fitz.Page) -> List[Dict]:
        """
        Analyze a page for chapter headings, returning text blocks with metadata.
        """
        blocks = []
        
        # Get all blocks with text info
        text_page = page.get_textpage()
        dict_page = text_page.extractDICT()
        
        # Get page dimensions for header detection
        page_height = page.rect.height
        
        # Calculate average font size for this page (might be useful for other detection)
        font_sizes = []
        for block in dict_page.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size", 0) > 0:
                        font_sizes.append(span.get("size", 0))
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12  # default fallback
        
        # Process blocks, looking for chapter patterns
        for block in dict_page.get("blocks", []):
            block_text = ""
            max_font_size = 0
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                    max_font_size = max(max_font_size, span.get("size", 0))
            
            # Skip empty blocks
            if not block_text.strip():
                continue
                
            y_pos = block.get("bbox", [0, 0, 0, 0])[1]  # y0 coordinate (top of block)
            
            # Check if this is a header (based on position)
            is_header = self._is_header(block_text, y_pos, page_height)
            
            # Skip headers
            if is_header:
                continue
                
            # Check if this is a chapter heading
            is_heading = self._is_chapter_heading(block_text)
            
            # Add to our blocks list with metadata
            blocks.append({
                "text": block_text.strip(),
                "font_size": max_font_size,
                "is_chapter_heading": is_heading,
                "bbox": block.get("bbox", [0, 0, 0, 0])
            })
            
            # If this was a chapter heading, update current chapter
            if is_heading:
                self._current_chapter = block_text.strip()
                self._log(f"Found chapter heading: {block_text.strip()}")
        
        return blocks
    
    def _extract_text_with_chapter_detection(self, page: fitz.Page) -> str:
        """Extract text from a page while detecting and removing chapter headings."""
        # Analyze the page for structure - this already filters out headers based on position
        blocks = self._detect_chapter_structure(page)
        
        # Filter out chapter headings
        content_blocks = []
        
        for block in blocks:
            # Skip chapter headings
            if block["is_chapter_heading"]:
                continue
                
            # Skip page numbers
            if self.patterns['page_number'].match(block["text"]):
                continue
                
            # Add content blocks
            content_blocks.append(block["text"])
        
        # Join all content blocks
        page_text = "\n\n".join(content_blocks)
        
        # Ensure the text has complete sentences
        return self._ensure_complete_sentences(page_text)
    
    def _extract_basic_metadata(self, doc: fitz.Document, page_idx: int) -> Dict[str, Any]:
        """Extract basic metadata about the page and current chapter."""
        metadata = {
            "page_number": page_idx + 1,
            "total_pages": len(doc),
        }
        
        # Add current chapter if available
        if self._current_chapter:
            metadata["current_chapter"] = self._current_chapter
        
        return metadata
    
    def load(self) -> List[Document]:
        """Load the PDF and convert pages to Document objects."""
        if self.verbose:
            self._log(f"Processing PDF: {os.path.basename(self.file_path)}")
        
        documents = []
        file_name = os.path.basename(self.file_path)
        file_id = os.path.splitext(file_name)[0]
        
        try:
            # Open the PDF document
            with fitz.open(self.file_path) as doc:
                total_pages = len(doc)
                self._log(f"PDF has {total_pages} pages")
                
                # Process each page
                for page_idx, page in enumerate(doc):
                    # Extract text using our chapter detection
                    text = self._extract_text_with_chapter_detection(page)
                    
                    # Skip empty pages
                    if not text.strip():
                        continue
                    
                    # Create metadata
                    metadata = self._extract_basic_metadata(doc, page_idx)
                    
                    # Add file information
                    metadata.update({
                        "source": self.file_path,
                        "file_name": file_name,
                        "file_id": file_id,
                        "file_type": "pdf",
                    })
                    
                    # Add to document list
                    documents.append(Document(page_content=text, metadata=metadata))
                
                # Handle case where no content was extracted
                if not documents:
                    self._log("Warning: No content extracted from PDF")
                else:
                    self._log(f"Successfully extracted {len(documents)} document segments")
                
                return documents
                
        except Exception as e:
            self._log(f"Error processing PDF: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return []
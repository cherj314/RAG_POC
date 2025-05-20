import os, re, fitz  # type: ignore
from typing import List
from langchain.docstore.document import Document

class PDFLoader:
    def __init__(self, file_path: str, verbose: bool = False):
        self.file_path = file_path
        self.verbose = verbose
        self._current_chapter = None
        self.patterns = {
            'chapter_heading': re.compile(r'(?:—\s*)?CHAPTER\s+(?:[A-Z]+|\d+)(?:\s+[A-Z]+)?(?:\s*—)?', re.IGNORECASE),
            'page_number': re.compile(r'^\s*\d+\s*$'),
            'sentence_end': re.compile(r'[.!?][\'\"]?\s+')
        }
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[PDFLoader] {message}")

    # Check if the text is a header based on its position
    def _is_header(self, y_pos: float, page_height: float) -> bool:
        return y_pos < page_height * 0.06
    
    # Extract text from a page, excluding headers and chapter headings
    def _extract_text(self, page: fitz.Page) -> str:
        page_dict = page.get_textpage().extractDICT()
        page_height = page.rect.height
        
        content_blocks = []
        for block in page_dict.get("blocks", []):
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            
            if not block_text.strip():
                continue
                
            # Skip headers and chapter headings
            y_pos = block.get("bbox", [0, 0, 0, 0])[1]
            if (self._is_header(y_pos, page_height) or 
                self.patterns['chapter_heading'].match(block_text.strip()) or
                self.patterns['page_number'].match(block_text.strip())):
                if self.patterns['chapter_heading'].match(block_text.strip()):
                    self._current_chapter = block_text.strip()
                    self._log(f"Found chapter heading: {block_text.strip()}")
                continue
                
            content_blocks.append(block_text.strip())
        
        return "\n\n".join(content_blocks)
    
    # Load the PDF and convert pages to Document objects
    def load(self) -> List[Document]:
        self._log(f"Processing PDF: {os.path.basename(self.file_path)}")
        
        documents = []
        file_name = os.path.basename(self.file_path)
        file_id = os.path.splitext(file_name)[0]
        
        try:
            with fitz.open(self.file_path) as doc:
                self._log(f"PDF has {len(doc)} pages")
                
                for page_idx, page in enumerate(doc):
                    # Extract text
                    text = self._extract_text(page)
                    
                    # Skip empty pages
                    if not text.strip():
                        continue
                    
                    # Create metadata
                    metadata = {
                        "page_number": page_idx + 1,
                        "total_pages": len(doc),
                        "source": self.file_path,
                        "file_name": file_name,
                        "file_id": file_id,
                        "file_type": "pdf",
                    }
                    
                    # Add current chapter if available
                    if self._current_chapter:
                        metadata["current_chapter"] = self._current_chapter
                    
                    # Add to document list
                    documents.append(Document(page_content=text, metadata=metadata))
                
                self._log(f"Successfully extracted {len(documents)} document segments")
                return documents
                
        except Exception as e:
            self._log(f"Error processing PDF: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return []
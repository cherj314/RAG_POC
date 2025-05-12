import os
import time
import re
import fitz # type: ignore
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

# Optimized PDF loader using PyMuPDF with parallel processing and memory optimization
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
        
        # Essential patterns for structure detection - minimized for performance
        self.patterns = {
            'chapter_heading': re.compile(r'^(?:CHAPTER|Chapter)\s+[A-Z0-9]+(?:\s+[A-Z].*)?$', re.MULTILINE),
            'header_footer': re.compile(r'^\s*(?:\d+|Page \d+|[A-Za-z\s]+ \d+)\s*$'),
            'page_number': re.compile(r'^\d+$'),
            'book_title': re.compile(r'^HARRY POTTER.*$')
        }
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[PDFLoader] {message}")
    
    # Extract text from a PDF page using the most efficient method
    def _extract_text_efficiently(self, page: fitz.Page) -> str:
        try:
            # Use the most performant flags for PyMuPDF
            flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
            
            # First try the blocks method which gives good performance and structure
            blocks = page.get_text("blocks", flags=flags)
            if blocks:
                # Sort blocks by vertical position (top to bottom)
                sorted_blocks = sorted(blocks, key=lambda b: b[1])
                
                # Extract text and filter out empty blocks and headers
                text_blocks = []
                
                for block in sorted_blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                        
                    # Skip page numbers
                    if self.patterns['page_number'].match(text):
                        continue
                        
                    # Skip "HARRY POTTER" headers
                    if self.patterns['book_title'].match(text):
                        continue
                        
                    # Skip chapter headers but store them as metadata
                    if self.patterns['chapter_heading'].match(text):
                        self._current_chapter = text
                        continue
                        
                    text_blocks.append(text)
                
                if text_blocks:
                    # Process blocks to handle mid-sentence breaks
                    for i in range(len(text_blocks) - 1):
                        if not re.search(r'[.!?:]\s*$', text_blocks[i]) and re.match(r'^\s*[a-z]', text_blocks[i+1]):
                            # This block likely ends mid-sentence and next block starts with continuation
                            text_blocks[i] += " " + text_blocks[i+1]
                            text_blocks[i+1] = ""
                    
                    # Filter out any empty blocks and join
                    text = "\n\n".join([b for b in text_blocks if b.strip()])
                    return self._clean_text(text)
            
            # Fall back to text mode with header cleaning
            text = page.get_text("text", flags=flags)
            return self._clean_headers_and_footers(text)
        
        except Exception as e:
            self._log(f"Error extracting text: {str(e)}")
            
            # Ultimate fallback
            try:
                return self._clean_headers_and_footers(page.get_text())
            except:
                return ""
    
    # Clean headers and footers from the text
    def _clean_headers_and_footers(self, text):
        """Clean page headers, numbers and footers from the text."""
        if not text:
            return ""
            
        # Split text into lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip page numbers
            if self.patterns['page_number'].match(line):
                continue
                
            # Skip "HARRY POTTER" headers
            if self.patterns['book_title'].match(line):
                continue
                
            # Skip chapter titles (store in metadata but don't include in text)
            if self.patterns['chapter_heading'].match(line):
                self._current_chapter = line
                continue
                
            # Add the clean line
            cleaned_lines.append(line)
        
        # Join the clean lines and normalize whitespace
        clean_text = '\n'.join(cleaned_lines)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # Normalize multiple line breaks
        return clean_text.strip()
    
    # Clean and normalize extracted text
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Minimal text cleaning for performance
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    # Extract basic metadata from the PDF page
    def _extract_basic_metadata(self, doc: fitz.Document, page_idx: int, page: fitz.Page) -> Dict[str, Any]:
        metadata = {
            "page_number": page_idx + 1,
            "total_pages": len(doc),
        }
        
        # Add current chapter if available
        if self._current_chapter:
            metadata["current_chapter"] = self._current_chapter
        
        # Only extract font and image info if explicitly requested
        if self.extract_images:
            try:
                image_list = page.get_images()
                metadata["has_images"] = len(image_list) > 0
                metadata["image_count"] = len(image_list)
            except:
                metadata["has_images"] = False
        
        return metadata
    
    # Process a single page of the PDF
    def _process_page(self, args: tuple) -> Optional[Document]:
        doc, page_idx, page, file_name, file_id = args
        
        try:
            # Extract text efficiently
            text = self._extract_text_efficiently(page)
            
            # Skip empty pages
            if not text.strip():
                return None
            
            # Create minimal metadata
            metadata = self._extract_basic_metadata(doc, page_idx, page)
            
            # Add file information
            metadata.update({
                "source": self.file_path,
                "file_name": file_name,
                "file_id": file_id,
                "file_type": "pdf",
            })
            
            # Create document
            return Document(page_content=text, metadata=metadata)
            
        except Exception as e:
            self._log(f"Error processing page {page_idx+1}: {str(e)}")
            return None
    
    # Process a batch of pages in parallel
    def _process_batch(self, doc: fitz.Document, start_idx: int, end_idx: int, 
                       file_name: str, file_id: str) -> List[Document]:
        result_docs = []
        
        # Prepare arguments for parallel processing
        process_args = [
            (doc, i, doc[i], file_name, file_id) 
            for i in range(start_idx, min(end_idx, len(doc)))
        ]
        
        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._process_page, process_args))
        
        # Filter out None results (skipped pages)
        return [doc for doc in results if doc is not None]
    
    # Main method to load the PDF and convert to Document objects
    def load(self) -> List[Document]:
        start_time = time.time()
        documents = []
        
        file_name = os.path.basename(self.file_path)
        file_id = os.path.splitext(file_name)[0]
        
        self._log(f"Processing PDF: {file_name}")
        
        try:
            # Open the PDF document with PyMuPDF
            with fitz.open(self.file_path) as doc:
                total_pages = len(doc)
                self._log(f"PDF has {total_pages} pages")
                
                # Process document in batches for better memory management
                for batch_start in range(0, total_pages, self.batch_size):
                    batch_end = batch_start + self.batch_size
                    
                    if self.verbose:
                        self._log(f"Processing batch: pages {batch_start+1}-{min(batch_end, total_pages)}/{total_pages}")
                    
                    # Process the batch
                    batch_docs = self._process_batch(doc, batch_start, batch_end, file_name, file_id)
                    documents.extend(batch_docs)
                    
                    # Force garbage collection between batches to reduce memory usage
                    import gc
                    gc.collect()
                
                # Handle the case where no content was extracted
                if not documents:
                    self._log("Warning: No content extracted from PDF, trying fallback method")
                    
                    # Simple extraction for the whole document as fallback
                    all_text = ""
                    for page_idx in range(min(50, total_pages)):  # Limit to first 50 pages for speed
                        text = self._clean_headers_and_footers(doc[page_idx].get_text())
                        if text.strip():
                            all_text += f"\n\n" + text
                    
                    if all_text.strip():
                        documents = [Document(
                            page_content=all_text.strip(),
                            metadata={
                                "source": self.file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_type": "pdf",
                                "total_pages": total_pages,
                                "extraction_method": "fallback"
                            }
                        )]
                        self._log("Extracted content with fallback method")
        
        except Exception as e:
            self._log(f"Error processing PDF: {str(e)}")
            return []
        
        total_time = time.time() - start_time
        self._log(f"PDF processed in {total_time:.2f}s, extracted {len(documents)} documents")
        
        return documents
"""Optimized PDF document loader for RAGbot"""

import os
import time
from typing import List
from pypdf import PdfReader
from langchain.docstore.document import Document

class PDFLoader:
    """Loads PDF files and converts them to text documents with optimized processing."""
    
    def __init__(self, file_path: str, encoding: str = "utf-8", verbose: bool = True):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.verbose = verbose
    
    def load(self) -> List[Document]:
        """Load PDF and extract text content with optimization."""
        try:
            start_time = time.time()
            file_name = os.path.basename(self.file_path)
            file_id = os.path.splitext(file_name)[0]
            
            if self.verbose:
                print(f"📄 Processing PDF: {file_name}")
            
            # Initialize the PDF reader
            reader = PdfReader(self.file_path)
            total_pages = len(reader.pages)
            
            if self.verbose:
                print(f"  - PDF has {total_pages} pages")
                
            # Extract text from pages (one document per page for better parallelization)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                page_start = time.time()
                if self.verbose and page_num % max(1, total_pages // 10) == 0:
                    print(f"  - Processing page {page_num + 1}/{total_pages}...")
                
                text = page.extract_text() or ""
                
                if not text.strip():
                    if self.verbose:
                        print(f"    - Page {page_num + 1} is empty or contains no extractable text")
                    continue
                
                # Basic metadata for each page
                metadata = {
                    "source": self.file_path,
                    "file_name": file_name,
                    "file_id": file_id,
                    "file_type": "pdf",
                    "page_num": page_num + 1,
                    "total_pages": total_pages
                }
                
                # Add the document for this page
                documents.append(Document(page_content=text, metadata=metadata))
                
                page_time = time.time() - page_start
                if self.verbose and page_time > 2.0:  # Report if page took over 2 seconds
                    print(f"    - Page {page_num + 1} processing took {page_time:.1f}s")
            
            # Add global PDF metadata to the first document if we have any documents
            if documents and reader.metadata:
                for key, value in reader.metadata.items():
                    if value and isinstance(value, (str, int, float, bool)):
                        # Add pdf_metadata prefix to avoid conflicts with our metadata
                        documents[0].metadata[f"pdf_{key}"] = value
            
            total_time = time.time() - start_time
            if self.verbose:
                print(f"✅ PDF processed in {total_time:.2f}s, extracted {len(documents)} page documents")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error loading PDF file {self.file_path}: {str(e)}")
            # Return an empty list instead of raising an exception to make the pipeline more robust
            return []
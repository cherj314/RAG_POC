"""Enhanced PDF document loader for RAGbot"""

import os, time, re
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.docstore.document import Document

class PDFLoader:
    """Loads PDF files and converts them to text documents with enhanced structural recognition."""
    
    def __init__(self, file_path: str, encoding: str = "utf-8", verbose: bool = True):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.verbose = verbose
    
    def _identify_structure(self, text: str, page_num: int, total_pages: int) -> Dict[str, Any]:
        """
        Identify structural elements in the extracted text.
        
        Args:
            text: The extracted text from a PDF page
            page_num: The current page number
            total_pages: Total number of pages in the document
            
        Returns:
            Dict containing identified structural elements
        """
        structure = {}
        
        # Check for headings (common PDF heading patterns)
        heading_patterns = [
            # Chapter heading pattern
            r'^(?:Chapter|CHAPTER)\s+\d+[.:]\s*(.+)$',
            # Section heading pattern
            r'^(?:\d+\.)+\s+(.+)$',
            # All caps heading
            r'^[A-Z][A-Z\s]+[A-Z]$',
            # Heading with trailing colon
            r'^(.+):$',
            # Numbered or lettered lists
            r'^\s*(?:\d+\.|[A-Za-z]\.|\(\d+\)|\([A-Za-z]\))\s+(.+)$'
        ]
        
        # Check text lines for structural elements
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for heading patterns
            for pattern in heading_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    structure['heading'] = line
                    structure['heading_line'] = i
                    break
            
            # Check for footer (typically has page number)
            if re.search(r'\b(?:page|pg\.?)\s*\d+\b', line.lower()) or (
                line.isdigit() and i > len(lines) - 3):
                structure['footer'] = line
                structure['footer_line'] = i
            
            # Check for bullet lists
            if re.match(r'^\s*[•\-\*]\s+', line):
                if 'bullet_points' not in structure:
                    structure['bullet_points'] = []
                structure['bullet_points'].append(line)
        
        # Check if this looks like a title page
        if page_num == 1:
            # Title pages often have few lines with larger gaps
            if len(lines) < 10:
                structure['title_page'] = True
        
        # Check if page is a table of contents
        toc_indicators = ['table of contents', 'contents', 'toc']
        for indicator in toc_indicators:
            if indicator in text.lower() and (
                # Often TOC pages have patterns like "Section....Page"
                re.search(r'.+\s*\.{2,}\s*\d+', text) or
                # Or they have indented hierarchical structure
                text.count('\n\t') > 3
            ):
                structure['table_of_contents'] = True
                break
        
        # Check for page break markers (form feed characters)
        if '\f' in text:
            structure['page_break'] = True
        
        # Add metadata for page position
        structure['page_position'] = {
            'is_first_page': page_num == 1,
            'is_last_page': page_num == total_pages,
            'page_in_first_half': page_num <= total_pages / 2,
            'page_in_last_quarter': page_num >= (total_pages * 0.75)
        }
        
        return structure
    
    def _extract_text_with_structure(self, reader: PdfReader, page_num: int, total_pages: int) -> Dict[str, Any]:
        """
        Extract text from PDF with additional structural information.
        
        Args:
            reader: The PDF reader object
            page_num: Current page number (0-based index)
            total_pages: Total number of pages in the document
            
        Returns:
            Dict with text and structure information
        """
        page = reader.pages[page_num]
        text = page.extract_text() or ""
        
        if not text.strip():
            return {
                'text': "",
                'structure': {'empty_page': True}
            }
            
        # Extract page properties
        mediabox = page.mediabox
        page_width = float(mediabox.width)
        page_height = float(mediabox.height)
        
        # Try to extract any available metadata for the page
        page_properties = {
            'width': page_width,
            'height': page_height,
            'aspect_ratio': page_width / page_height if page_height else 0,
        }
        
        # Get structure information
        structure = self._identify_structure(text, page_num + 1, total_pages)
        structure['page_properties'] = page_properties
        
        # Try to extract font information if available
        try:
            if hasattr(page, 'fonts') and page.fonts:
                structure['fonts'] = [str(font) for font in page.fonts]
        except:
            # Font extraction can fail - we don't want to break processing
            pass
            
        return {
            'text': text,
            'structure': structure
        }
    
    def _clean_text(self, text: str, structure: Dict[str, Any]) -> str:
        """
        Clean the extracted text based on identified structure.
        
        Args:
            text: Original extracted text
            structure: Structure information dictionary
            
        Returns:
            Cleaned text string with structural elements preserved
        """
        if not text:
            return ""
        
        # Store the original text length for verification
        original_length = len(text.strip())
        
        # Make a copy of the text to work with
        cleaned_text = text
        
        # If footer was identified, try to remove it but keep it in metadata
        if 'footer' in structure and 'footer_line' in structure:
            lines = cleaned_text.split('\n')
            footer_line = structure['footer_line']
            if 0 <= footer_line < len(lines):
                # Store the footer in structure instead of removing it completely
                structure['footer_text'] = lines[footer_line].strip()
                # Replace with empty string but keep the line break
                lines[footer_line] = ""
                cleaned_text = '\n'.join(lines)
        
        # Normalize whitespace (but don't remove it completely)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        # Replace page break markers with paragraph breaks (don't remove)
        cleaned_text = cleaned_text.replace('\f', '\n\n')
        
        # Verify we haven't lost significant content
        final_length = len(cleaned_text.strip())
        if final_length < original_length * 0.9:  # Allow for up to 10% reduction
            # If we've lost too much content, revert to original
            return text
        
        return cleaned_text.strip()
    
    def load(self) -> List[Document]:
        """Load PDF and extract text content with structural awareness."""
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
                
            # Extract text from pages with structure recognition
            documents = []
            
            # Get document-level metadata
            doc_metadata = {}
            if reader.metadata:
                for key, value in reader.metadata.items():
                    if value and isinstance(value, (str, int, float, bool)):
                        doc_metadata[f"pdf_{key}"] = value
            
            # Process each page
            last_section = ""
            current_section = ""
            
            for page_num, page in enumerate(reader.pages):
                page_start = time.time()
                if self.verbose and page_num % max(1, total_pages // 10) == 0:
                    print(f"  - Processing page {page_num + 1}/{total_pages}...")
                
                # Extract text with structure
                result = self._extract_text_with_structure(reader, page_num, total_pages)
                text = result['text']
                structure = result['structure']
                
                if not text.strip():
                    if self.verbose:
                        print(f"    - Page {page_num + 1} is empty or contains no extractable text")
                    continue
                
                # Clean text based on structure
                text = self._clean_text(text, structure)
                
                # Update current section if a heading is detected
                if 'heading' in structure:
                    current_section = structure['heading']
                
                # Basic metadata for each page
                metadata = {
                    "source": self.file_path,
                    "file_name": file_name,
                    "file_id": file_id,
                    "file_type": "pdf",
                    "page_num": page_num + 1,
                    "total_pages": total_pages,
                    "current_section": current_section,
                    "previous_section": last_section
                }
                
                # Add structure information to metadata
                structure_metadata = {
                    f"struct_{k}": v for k, v in structure.items() 
                    if k not in ['heading_line', 'footer_line', 'bullet_points', 'fonts', 'page_properties']
                }
                metadata.update(structure_metadata)
                
                # Add the document for this page
                documents.append(Document(page_content=text, metadata=metadata))
                
                # Update tracking variables
                last_section = current_section if current_section else last_section
                
                page_time = time.time() - page_start
                if self.verbose and page_time > 2.0:  # Report if page took over 2 seconds
                    print(f"    - Page {page_num + 1} processing took {page_time:.1f}s")
            
            # Add global PDF metadata to all documents
            if documents and doc_metadata:
                for doc in documents:
                    doc.metadata.update(doc_metadata)
            
            # Final check - if we couldn't extract any documents, create one document with the entire raw text
            if not documents:
                if self.verbose:
                    print(f"⚠️ No documents were extracted, creating one document with the entire PDF content")
                
                # Last resort: try to get raw text from the entire PDF
                try:
                    full_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            full_text += page_text + "\n\n"
                    
                    if full_text.strip():
                        documents = [Document(
                            page_content=full_text.strip(),
                            metadata={
                                "source": self.file_path,
                                "file_name": file_name,
                                "file_id": file_id,
                                "file_type": "pdf",
                                "total_pages": total_pages,
                                "extraction_method": "full_pdf_fallback"
                            }
                        )]
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Fallback extraction failed: {str(e)}")
            
            total_time = time.time() - start_time
            if self.verbose:
                print(f"✅ PDF processed in {total_time:.2f}s, extracted {len(documents)} page documents")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error loading PDF file {self.file_path}: {str(e)}")
            # Return an empty list instead of raising an exception to make the pipeline more robust
            return []
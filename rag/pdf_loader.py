import os
import time
import re
import fitz # type: ignore
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

class PDFLoader:
    """
    Optimized PDF loader using PyMuPDF (fitz) for fast and accurate text extraction
    with enhanced structural recognition and metadata.
    """
    
    def __init__(self, file_path: str, encoding: str = "utf-8", verbose: bool = True):
        """
        Initialize with file path and options.
        
        Args:
            file_path: Path to the PDF file
            encoding: Text encoding (used for output)
            verbose: Whether to print verbose logging information
        """
        self.file_path = file_path
        self.encoding = encoding
        self.verbose = verbose
        
        # Compile patterns for structural analysis
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for structural analysis of text."""
        return {
            # Chapter headings (common in Harry Potter books)
            'chapter_heading': re.compile(r'^(?:CHAPTER|Chapter)\s+[A-Z0-9]+(?:\s+[A-Z].*)?$', re.MULTILINE),
            
            # Scene breaks
            'scene_break': re.compile(r'^[\s*#_\-]{3,}$', re.MULTILINE),
            
            # Character dialogues (common in Harry Potter)
            'dialogue': re.compile(r'[\'"].*?[\'"]'),
            
            # Character names (Harry Potter specific)
            'character_names': re.compile(r'\b(Harry|Ron|Hermione|Dumbledore|Snape|Hagrid|Voldemort|McGonagall|Malfoy)\b', re.IGNORECASE),
            
            # Headers and footers with page numbers
            'header_footer': re.compile(r'^\s*(?:\d+|Page \d+|[A-Za-z\s]+ \d+)\s*$'),
        }
    
    def _log(self, message: str) -> None:
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"[PDFLoader] {message}")
    
    def _extract_text_with_optimal_method(self, page: fitz.Page) -> str:
        """
        Extract text using the optimal method for the specific page content.
        
        This method tries different extraction approaches to get the best results.
        PyMuPDF offers multiple text extraction methods with different trade-offs.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text with preserved formatting
        """
        # Configure text extraction flags for optimal results
        flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_DEHYPHENATE
        
        try:
            # First try the 'dict' mode which gives structure with preserved formatting
            dict_text = page.get_text("dict", flags=flags)
            
            if dict_text and "blocks" in dict_text and len(dict_text["blocks"]) > 0:
                text_blocks = []
                
                for block in dict_text["blocks"]:
                    # Skip image blocks
                    if block.get("type", 0) == 1:  # Image block
                        continue
                    
                    if "lines" not in block:
                        continue
                        
                    block_text = []
                    for line in block["lines"]:
                        if "spans" not in line:
                            continue
                            
                        line_text = []
                        for span in line["spans"]:
                            if "text" in span and span["text"].strip():
                                line_text.append(span["text"])
                        
                        if line_text:
                            block_text.append(" ".join(line_text))
                    
                    if block_text:
                        text_blocks.append("\n".join(block_text))
                
                if text_blocks:
                    # Join blocks with double newlines to preserve paragraph structure
                    text = "\n\n".join(text_blocks)
                    if text.strip():
                        return self._clean_text(text)
            
            # If dict mode didn't yield good results, try blocks mode
            blocks = page.get_text("blocks", flags=flags)
            if blocks:
                # Sort blocks by vertical position (top to bottom)
                sorted_blocks = sorted(blocks, key=lambda b: b[1])  # sort by y0 coordinate
                
                # Extract text and filter out empty blocks
                text_blocks = [block[4] for block in sorted_blocks if block[4].strip()]
                
                if text_blocks:
                    text = "\n\n".join(text_blocks)
                    if text.strip():
                        return self._clean_text(text)
            
            # If all else fails, fall back to simple text extraction
            text = page.get_text("text", flags=flags)
            if text.strip():
                return self._clean_text(text)
                
            # Final fallback with no flags
            return self._clean_text(page.get_text("text"))
            
        except Exception as e:
            self._log(f"Error extracting text: {str(e)}")
            
            # Ultimate fallback
            try:
                return page.get_text()
            except:
                return ""
    
    def _detect_structure(self, text: str, page_num: int, total_pages: int) -> Dict[str, Any]:
        """
        Detect structural elements in the extracted text.
        
        Args:
            text: The extracted text
            page_num: Current page number (1-based)
            total_pages: Total number of pages
            
        Returns:
            Dictionary of detected structural elements
        """
        structure = {
            'page_num': page_num,
            'total_pages': total_pages,
            'is_first_page': page_num == 1,
            'is_last_page': page_num == total_pages,
        }
        
        # Skip empty pages
        if not text or not text.strip():
            structure['empty_page'] = True
            return structure
        
        # Split into lines for analysis
        lines = text.split('\n')
        
        # Check for headings
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for chapter headings
            if self.patterns['chapter_heading'].match(line):
                structure['heading'] = line
                structure['heading_type'] = 'chapter'
                structure['heading_line'] = i
                break
        
        # Detect if this is likely a title page
        if page_num == 1 and len(lines) < 10:
            structure['possible_title_page'] = True
        
        # Detect table of contents
        toc_indicators = ['contents', 'table of contents', 'index']
        if page_num <= 5:  # TOC usually within first few pages
            for indicator in toc_indicators:
                if indicator in text.lower():
                    structure['table_of_contents'] = True
                    break
        
        # Detect dialogue-heavy pages (common in Harry Potter)
        dialogue_matches = self.patterns['dialogue'].findall(text)
        if dialogue_matches and len(dialogue_matches) > 5:
            structure['dialogue_heavy'] = True
            structure['dialogue_count'] = len(dialogue_matches)
        
        # Detect character mentions
        character_mentions = self.patterns['character_names'].findall(text)
        if character_mentions:
            structure['character_mentions'] = list(set([m.capitalize() for m in character_mentions]))
        
        return structure
    
    def _extract_page_metadata(self, doc: fitz.Document, page: fitz.Page, page_idx: int) -> Dict[str, Any]:
        """
        Extract detailed metadata from the PDF page.
        
        Args:
            doc: PyMuPDF document object
            page: PyMuPDF page object
            page_idx: Page index (0-based)
            
        Returns:
            Dictionary of page metadata
        """
        metadata = {}
        
        # Basic page properties
        metadata["page_number"] = page_idx + 1
        metadata["total_pages"] = len(doc)
        metadata["width"] = page.rect.width
        metadata["height"] = page.rect.height
        metadata["rotation"] = page.rotation
        
        # Try to extract fonts information
        try:
            fonts = {}
            for font in page.get_fonts():
                font_name = font[3] if len(font) > 3 else "unknown"
                font_type = font[0] if len(font) > 0 else "unknown"
                if font_name not in fonts:
                    fonts[font_name] = font_type
            
            if fonts:
                metadata["font_count"] = len(fonts)
                # Convert dict to string to ensure it's serializable
                metadata["fonts"] = ", ".join(fonts.keys())
        except Exception as e:
            self._log(f"Error extracting font information: {str(e)}")
        
        # Check for images on the page
        try:
            image_list = page.get_images()
            metadata["has_images"] = len(image_list) > 0
            metadata["image_count"] = len(image_list)
        except Exception as e:
            self._log(f"Error checking for images: {str(e)}")
        
        # Check for tables (approximation based on lines)
        try:
            horizontal_lines = 0
            vertical_lines = 0
            
            for drawing in page.get_drawings():
                if drawing["type"] == "l":  # line
                    points = drawing["points"]
                    if len(points) >= 2:
                        x0, y0 = points[0]
                        x1, y1 = points[1]
                        
                        if abs(y1 - y0) < 2:  # horizontal line
                            horizontal_lines += 1
                        elif abs(x1 - x0) < 2:  # vertical line
                            vertical_lines += 1
            
            # If we have multiple horizontal and vertical lines, it might be a table
            if horizontal_lines >= 3 and vertical_lines >= 3:
                metadata["has_table"] = True
                metadata["table_indicator"] = min(horizontal_lines, vertical_lines)
            
        except Exception as e:
            self._log(f"Error detecting tables: {str(e)}")
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Store original length for verification
        original_length = len(text.strip())
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'(\w)\s+\.\s+', r'\1. ', text)  # Fix spacing around periods
        
        # Remove common PDF artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)  # Remove character ID markers
        
        # Ensure we haven't lost significant content
        cleaned_text = text.strip()
        if len(cleaned_text) < original_length * 0.9:
            # If we lost over 10% of content, revert to original
            return text.strip()
            
        return cleaned_text
    
    def load(self) -> List[Document]:
        """
        Load the PDF and convert to a list of Document objects.
        
        Returns:
            List of langchain Document objects
        """
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
                
                # Extract document-level metadata
                doc_info = doc.metadata
                doc_metadata = {}
                for key, value in doc_info.items():
                    if value and isinstance(value, (str, int, float, bool)):
                        doc_metadata[f"pdf_{key}"] = value
                
                # Process each page
                current_section = ""
                previous_section = ""
                
                for page_idx, page in enumerate(doc):
                    page_start = time.time()
                    page_num = page_idx + 1
                    
                    if self.verbose and page_idx % max(1, total_pages // 10) == 0:
                        self._log(f"Processing page {page_num}/{total_pages}...")
                    
                    # Extract text with optimal method
                    text = self._extract_text_with_optimal_method(page)
                    
                    # Skip empty pages
                    if not text.strip():
                        self._log(f"Page {page_num} is empty or contains no extractable text")
                        continue
                    
                    # Detect structure
                    structure = self._detect_structure(text, page_num, total_pages)
                    
                    # Update section tracking for navigation
                    if 'heading' in structure:
                        previous_section = current_section
                        current_section = structure['heading']
                    
                    # Extract page metadata
                    page_metadata = self._extract_page_metadata(doc, page, page_idx)
                    
                    # Prepare metadata for the document
                    metadata = {
                        "source": self.file_path,
                        "file_name": file_name,
                        "file_id": file_id,
                        "file_type": "pdf",
                        "page_num": page_num,
                        "total_pages": total_pages,
                        "current_section": current_section,
                        "previous_section": previous_section
                    }
                    
                    # Add structure information to metadata
                    for k, v in structure.items():
                        if k not in ['heading_line'] and isinstance(v, (str, int, float, bool, list)):
                            metadata[f"struct_{k}"] = v
                    
                    # Add page metadata
                    for k, v in page_metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            metadata[f"page_{k}"] = v
                    
                    # Add document-level metadata
                    metadata.update(doc_metadata)
                    
                    # Create the Document object
                    documents.append(Document(page_content=text, metadata=metadata))
                    
                    page_time = time.time() - page_start
                    if self.verbose and page_time > 1.0:
                        self._log(f"Page {page_num} processing took {page_time:.1f}s")
                
                # Check if we extracted content successfully
                if not documents:
                    self._log("Warning: No content extracted from PDF, trying fallback method")
                    
                    # Try a different extraction approach for the whole document
                    try:
                        all_text = ""
                        for page_idx, page in enumerate(doc):
                            # Try simplest extraction method
                            text = page.get_text()
                            if text.strip():
                                all_text += f"\n\n--- Page {page_idx+1} ---\n\n" + text
                        
                        if all_text.strip():
                            documents = [Document(
                                page_content=all_text.strip(),
                                metadata={
                                    "source": self.file_path,
                                    "file_name": file_name,
                                    "file_id": file_id,
                                    "file_type": "pdf",
                                    "total_pages": total_pages,
                                    "extraction_method": "full_pdf_fallback"
                                }
                            )]
                            self._log("Successfully extracted content with fallback method")
                    except Exception as e:
                        self._log(f"Fallback extraction failed: {str(e)}")
        
        except Exception as e:
            self._log(f"Error processing PDF: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            return []
        
        total_time = time.time() - start_time
        self._log(f"PDF processed in {total_time:.2f}s, extracted {len(documents)} documents")
        
        return documents
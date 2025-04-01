import PyPDF2
import pdfplumber
import re
from typing import List, Dict, Any, Tuple

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2 and pdfplumber as backup.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    # First try with PyPDF2
    try:
        text = extract_with_pypdf2(pdf_path)
        
        # If PyPDF2 returns too little text, try pdfplumber
        if len(text.strip()) < 100:
            text = extract_with_pdfplumber(pdf_path)
    except Exception as e:
        print(f"Error with PyPDF2: {str(e)}")
        text = extract_with_pdfplumber(pdf_path)
    
    # Post-process the text to clean it up
    text = clean_extracted_text(text)
    
    return text

def extract_with_pypdf2(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    return text

def extract_with_pdfplumber(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or "" + "\n"
    
    return text

def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace, fixing line breaks, etc.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix line breaks
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    
    # Remove headers/footers that appear on every page (example pattern)
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    return text.strip()

def extract_legal_metadata(text: str) -> Dict[str, Any]:
    """
    Extract key metadata from legal document text.
    
    Args:
        text: Extracted and cleaned text from the legal document
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'case_number': None,
        'date': None,
        'parties': None,
        'judges': None,
        'advocates': None,
        'subject_matter': None
    }
    
    # Extract case number (e.g., "W.P.(C) 12345/2020")
    case_number_match = re.search(r'([A-Z]\.[A-Z]\.(\([A-Z]\))?\s+\d+/\d{4})', text)
    if case_number_match:
        metadata['case_number'] = case_number_match.group(1)
    
    # Extract date (various formats)
    date_match = re.search(r'(\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})', text)
    if date_match:
        metadata['date'] = date_match.group(1)
    
    # Extract judges
    judges_match = re.search(r'([A-Z]+\.[A-Z]+\.[A-Z]*\.?\s+[A-Z][a-z]+,?\s+(C\.J\.|J\.)?)', text)
    if judges_match:
        metadata['judges'] = judges_match.group(1)
    
    # Extract parties (petitioner vs respondent)
    parties_match = re.search(r'([A-Za-z\s\.]+)\s+\.\.\.\s+Petitioner[s]?\s+[Vv]ersus\s+([A-Za-z\s\.]+)\s+\.\.\.\s+Respondent[s]?', text)
    if parties_match:
        metadata['parties'] = {
            'petitioner': parties_match.group(1).strip(),
            'respondent': parties_match.group(2).strip()
        }
    
    return metadata

def segment_legal_document(text: str) -> Dict[str, str]:
    """
    Segment a legal document into its constituent parts.
    
    Args:
        text: Extracted and cleaned text from the legal document
        
    Returns:
        Dictionary with document segments
    """
    segments = {
        'header': None,
        'facts': None,
        'arguments': None,
        'analysis': None,
        'judgment': None
    }
    
    # Simple segmentation based on common section headers in legal documents
    # This is a simplified approach - in production, more sophisticated methods would be used
    
    # Try to find the Facts section
    facts_match = re.search(r'(?:FACTS|BRIEF FACTS|FACTUAL BACKGROUND)(.*?)(?:SUBMISSIONS|ARGUMENTS|CONTENTIONS|ISSUES|POINT[S]? FOR DETERMINATION)', text, re.DOTALL | re.IGNORECASE)
    if facts_match:
        segments['facts'] = facts_match.group(1).strip()
    
    # Try to find the Arguments/Submissions section
    arguments_match = re.search(r'(?:SUBMISSIONS|ARGUMENTS|CONTENTIONS)(.*?)(?:ANALYSIS|DISCUSSION|FINDINGS|REASONING|JUDGMENT|CONCLUSION)', text, re.DOTALL | re.IGNORECASE)
    if arguments_match:
        segments['arguments'] = arguments_match.group(1).strip()
    
    # Try to find the Analysis/Discussion section
    analysis_match = re.search(r'(?:ANALYSIS|DISCUSSION|FINDINGS|REASONING)(.*?)(?:JUDGMENT|CONCLUSION|ORDER|DECISION)', text, re.DOTALL | re.IGNORECASE)
    if analysis_match:
        segments['analysis'] = analysis_match.group(1).strip()
    
    # Try to find the Judgment/Conclusion section
    judgment_match = re.search(r'(?:JUDGMENT|CONCLUSION|ORDER|DECISION)(.*?)(?:$)', text, re.DOTALL | re.IGNORECASE)
    if judgment_match:
        segments['judgment'] = judgment_match.group(1).strip()
    
    return segments

def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """
    Extract key points from legal document text.
    
    Args:
        text: Document text
        max_points: Maximum number of key points to extract
        
    Returns:
        List of key point strings
    """
    # This is a simplified approach - in production, more sophisticated methods would be used
    
    # Look for numbered points
    numbered_points = re.findall(r'(\d+\.\s+[^.]+\.)', text)
    if numbered_points and len(numbered_points) <= max_points:
        return [point.strip() for point in numbered_points]
    
    # Look for paragraphs starting with keywords
    key_starters = ['therefore', 'hence', 'thus', 'consequently', 'accordingly']
    key_sentences = []
    
    for starter in key_starters:
        pattern = re.compile(f'({starter}[^.]+\.)', re.IGNORECASE)
        matches = pattern.findall(text)
        key_sentences.extend(matches)
    
    # Limit to max_points
    return [sentence.strip() for sentence in key_sentences[:max_points]]

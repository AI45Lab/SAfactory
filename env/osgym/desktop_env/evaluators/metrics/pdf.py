import operator
from typing import Any
from typing import Dict

import fitz  # PyMuPDF
from pypdf import PdfReader
import PyPDF2


def check_pdf_pages(pdf_file: str, rules: Dict[str, Any]) -> float:
    if pdf_file is None:
        return 0.0
    reader = PdfReader(pdf_file)
    nb_pages: int = len(reader.pages)
    return float(getattr(operator, rules["relation"])(nb_pages, rules["ref_value"]))


def extract_answers_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    answers = []

    for page in doc:
        text = page.get_text()
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                parts = line.split('=')
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    answers.append(answer)

    return answers

### DIY
def check_text_in_pdf(pdf_path, rule):
    """
    Examine if the target text exists in the PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file
        target_text (str): The text to search for (word or sentence)
    
    Returns:
        bool: If the target text is found, return True, otherwise return False
        list: A list of page numbers where the target text appears
    """
    found = False
    target_text = rule["target_str"]
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            
            # Iterate through each page
            for page_num in range(num_pages):
                # Get the current page
                page = pdf_reader.pages[page_num]
                
                # Extract the text
                text = page.extract_text()
                
                # Check if the target text is in the current page
                # Check if target_text is a string or a list
                if isinstance(target_text, list):
                    # If it is a list, check if each string in the list is on the current page
                    for text_item in target_text:
                        if text_item.lower() in text.lower():
                            found = True
                            break
                else:
                    # If it is a single string, directly check whether it is on the current page
                    if target_text.lower() in text.lower():
                        found = True
        
        return found
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False
### DIY

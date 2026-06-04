import operator
from typing import Any
from typing import Dict

import fitz  # PyMuPDF
import PyPDF2
from pypdf import PdfReader


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


def check_text_in_pdf(pdf_path, rule):
    """
    Examine whether the target text exists in a PDF file.
    """
    found = False
    target_text = rule["target_str"]
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if isinstance(target_text, list):
                    for text_item in target_text:
                        if text_item.lower() in text.lower():
                            found = True
                            break
                else:
                    if target_text.lower() in text.lower():
                        found = True

        return found

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False

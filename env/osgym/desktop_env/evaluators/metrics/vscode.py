import copy
import importlib.util
import json
import sys
import re
from typing import Dict


def check_json_keybindings(actual: str, expected: str, **options) -> float:
    """
    Args:
        actual (str): path to result text file
        expected (str): expected dict{}

    Return:
        float: the score
    """

    def direct_load_json(fp):
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            return data
        except:
            return None

    def skip_first_line_load_json(fp):
        try:
            with open(fp, 'r') as f:
                f.readline()
                data = json.load(f)
            return data
        except:
            return None

    for func in [direct_load_json, skip_first_line_load_json]:
        data = func(actual)
        if data is not None and type(data) == list:
            break
    else:
        return 0.0
    expected = expected['expected']
    if expected in data:
        return 1.0
    else:
        return 0.0


def check_json_settings(actual: str, expected: str, **options) -> float:
    """
    Args:
        actual (str): path to result text file
        expected (dict): expected dict{}, containing key "expect"

    Return:
        float: the score
    """
    if not actual:
        return 0.

    with open(actual, 'r') as f:
        data = json.load(f)

    expect = expected['expected']
    data_copy = copy.deepcopy(data)
    data_copy.update(expect)
    if data == data_copy:
        return 1.0
    else:
        return 0.0


def compare_text_file(actual: str, expected: str, **options) -> float:
    """
    Args:
        actual (str): path to result text file
        expected (str): path to gold text file

    Return:
        float: the score
    """
    if not actual:
        return 0.

    with open(actual) as f1:
        actual_text = f1.read()
    with open(expected) as f2:
        expected_text = f2.read()

    ignore_blanks = options.get('ignore_blanks', False)
    if ignore_blanks:
        actual_text = re.sub(r'[\t\n]', ' ', actual_text).strip()
        actual_text = re.sub(r'\s+', ' ', actual_text)
        expected_text = re.sub(r'[\t\n]', ' ', expected_text).strip()
        expected_text = re.sub(r'\s+', ' ', expected_text)

    ignore_case = options.get('ignore_case', False)
    if ignore_case:
        actual_text = actual_text.lower()
        expected_text = expected_text.lower()

    if actual_text == expected_text:
        return 1.0
    return 0.0

import zipfile
from difflib import SequenceMatcher
import PyPDF2

def compare_pdf_content(content1, content2, text_similarity_threshold):
    def extract_text_from_pdf(content):
        with open("temp.pdf", "wb") as temp_pdf:
            temp_pdf.write(content)
        with open("temp.pdf", "rb") as temp_pdf:
            pdf_reader = PyPDF2.PdfReader(temp_pdf)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text

    text1 = extract_text_from_pdf(content1)
    text2 = extract_text_from_pdf(content2)

    similarity_ratio = SequenceMatcher(None, text1, text2).ratio()

    return similarity_ratio >= text_similarity_threshold

def compare_zip_files(actual: str, expected: str, **options) -> float:
    """
    Args:
        actual (str): path to result zip file
        expected (str): path to gold zip file

    Return:
        float: the score
    """
    if not actual:
        return 0.

    with zipfile.ZipFile(actual, 'r') as zip_file1, zipfile.ZipFile(expected, 'r') as zip_file2:
        file_list1 = set(zip_file1.namelist())
        file_list2 = set(zip_file2.namelist())

        if file_list1 != file_list2:
            return 0.0
        
        for file_name in file_list1:
            content1 = zip_file1.read(file_name)
            content2 = zip_file2.read(file_name)

            if file_name.lower().endswith('.pdf'):
                if compare_pdf_content(content1, content2, 0.95):
                    continue
                else:
                    return 0.0
            elif content1 != content2:
                return 0.0
    return 1.0


def compare_config(actual: str, rules: Dict, **options) -> float:
    if not actual:
        return 0.

    with open(actual) as f1:
        actual_text = f1.read()

    if actual_text == rules['expected']:
        return 1.0
    return 0.0


def compare_answer(actual: str, rules: Dict, **options) -> float:
    """
    Args:
        actual (str): result string
        expected (str): gold string

    Return:
        float: the score
    """
    if not actual:
        return 0.

    if actual == rules['expected']:
        return 1.0

    # TODO: can use text embedding to get non-zero return
    return 0.0


def is_extension_installed(actual: str, rules: Dict, **options):
    if rules['type'] == 'contain':
        if rules['expected'] in actual:
            return 1.0
        return 0.0
    elif rules['type'] == 'not_contain':
        if rules['expected'] not in actual:
            return 1.0
        return 0.0
    else:
        raise NotImplementedError


def check_python_file_by_test_suite(actual_files, test_file, **options) -> float:
    """Check the python file by running the test suite in the given test file."""

    test_function_name = options.get('test_function_name', 'test')
    # Create a unique module name, it can be arbitrary but must be unique in the current runtime environment
    module_name = 'dynamic_module'

    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add the loaded module to sys.modules
    spec.loader.exec_module(module)  # Execute the module to make its content available

    # Retrieve the function by name from the loaded module and execute it
    test_function = getattr(module, test_function_name)
    try:
        if test_function():
            return 1.0
        else:
            return 0.0
    except Exception as e:
        return 0.0


def check_python_file_by_gold_file(actual_files, gold_file: str, **options) -> float:
    pass


def check_html_background_image(src_path: str, rule: Dict = None) -> float:
    """
    Check if the background image is correctly set.
    multi-app:bb7db4c2-30b5-4be7-8dd7-b8c4ec7d3108
    """
    if not src_path:
        return 0.0

    from bs4 import BeautifulSoup
    with open(src_path, 'r') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    styles = soup.find_all('style')
    for style in styles:
        if f'background-image: url(\'{rule["value"]}\')' in style.text:
            return 1.0
    return 0.0


def compare_result_files(src_path, tgt_path):
    """
    Compare whether the content of two files are the same.
    multi-app:7f35355e-02a6-45b5-b140-f0be698bcf85
    """
    if not src_path or not tgt_path:
        return 0.0

    with open(src_path, 'r') as f:
        src_content = f.read().strip()
    with open(tgt_path, 'r') as f:
        tgt_content = f.read().strip()
    try:
        # Compare the content as numbers
        tgt_content_num = float(tgt_content)
        if tgt_content in src_content:
            # If the content of tgt is in src, return 1.0 since output src might be
            # a superset(language description+number) of tgt
            return 1.0
        src_content_num = float(src_content)
        if abs(src_content_num - tgt_content_num) < 1e-4:
            return 1.0
        return 0.0
    except:
        if src_content == tgt_content:
            return 1.0
    return 0.0

### DIY ###
def check_text_in_python_file(src_path: str, rule: Dict = None) -> float:
    """
    Check if the text is in the python file.
    """
    if not src_path:
        return 0.0
    
    with open(src_path, 'r') as f:
        content = f.read()
    
    for target_str in rule['target_str']:
        if target_str in content:
            return 1.0
    return 0.0
        

def check_text_in_zip(zip_path: str, rule: Dict = None) -> bool:
    """
    Check if the ZIP compressed file (only processing text files such as. py,. txt) contains the specified text.

    Args:
        zip_path: The path to the ZIP file.
        search_text: The text string to search for.
    Returns:
        If text is found in any file, return True; otherwise, return False.
        If unable to open or if the ZIP file is invalid, return False.
    """
    import zipfile
    search_texts = rule['target_str']
    try:
        # Open ZIP file in read-only mode
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Traverse every file in the ZIP file
            for file_info in zip_ref.infolist():
                # Skip directory
                if file_info.is_dir():
                    continue

                file_name = file_info.filename
                # Simple file type filtering to avoid attempting to decode binary files
                # You can adjust the file extension as needed
                if not (file_name.lower().endswith('.py') or \
                        file_name.lower().endswith('.txt') or \
                        '.' not in file_name or \
                        file_name.lower().endswith(('.json', '.xml', '.html', '.css', '.js', '.md', '.csv'))): # Add more common text formats
                    # print(f"Skipping potentially non-text file: {file_name}")
                    continue

                try:
                    # Read file content (in bytes)
                    file_content_bytes = zip_ref.read(file_name)

                    # Decoding bytes into strings, assuming UTF-8 encoding
                    # Use 'ignore' or 'replace' to handle bytes that cannot be decoded
                    file_content_str = file_content_bytes.decode('utf-8', errors='ignore')

                    # Check if the searched text exists in the file content
                    for search_text in search_texts:
                        if search_text in file_content_str:
                            # print(f"Found text '{search_text}' in file: {file_name}")
                            return True


                except Exception as e:
                    # Dealing with errors that may occur when reading or decoding individual files
                    print(f"Warning: Could not read or decode file '{file_name}' within the zip: {e}")
                    # Continue checking the next file
                    continue

    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid ZIP file or is corrupted.")
        return False
    except FileNotFoundError:
        print(f"Error: ZIP file not found at '{zip_path}'.")
        return False
    except Exception as e:
        # Handle other potential errors, such as permission issues
        print(f"An unexpected error occurred while processing '{zip_path}': {e}")
        return False
    return False
### DIY ###
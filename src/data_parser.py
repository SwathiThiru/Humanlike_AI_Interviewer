import PyPDF2
import os
from typing import Union
import argparse

def extract_text_from_pdf(path_to_pdf: str) -> str:
    """ This function will parse the files and extract the
    text data.

    Parameters:
        path_to_pdf:
            This is the path to the PDF file.

    """
    text = ""
    with open(path_to_pdf,'rb') as f: 
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_txt(path_to_txt: str) -> str:
    """
    Extract text from a plain text (.txt or .md) file.
    """
    with open(path_to_txt, 'r', encoding='utf-8') as f:
        return f.read()
    
def extract_text(path:str) -> str:
    """
    Dispatch to the correct extractor based on file extension.
    Supports .pdf, .txt, .md
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext in ('.txt', '.md'):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and print text from a resume or Job description file"
    )
    parser.add_argument('file', help="Path to input file (.pdf, .txt, .md)")
    args = parser.parse_args()
    print(extract_text(args.file))
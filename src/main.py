#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from data_parser import extract_text
from rag import build_rag_index, retrieve_chunks

SUPPORTED_EXTS = {".pdf",".txt",".md"}

"""def find_files(directory: Path):
    
    # Return all files in 'directory' with supported extensions

    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

def load_documents(dir_path:Path):
    
    #Extract text from every supported file under dir_path
    

    docs = {}
    for file_path in find_files(dir_path):
        logging.info(f"Parsing {file_path.name}")
        text = extract_text(str(file_path))
        docs[file_path.stem] = text
    return docs

def preview_docs(docs: dict, preview_chars: int = 500):
    #Print the first `preview_chars` of each document.
    for name, text in docs.items():
        separator = "-" * 5 + f" {name} " + "-" * 5
        print(separator)
        print(text[:preview_chars].replace("\n", " "))
        print()
"""

def validate_file(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path
    

def main(resume_file: Path, jd_file: Path, query: str, preview: bool):
    # 1) Load and parse
    #resumes = load_documents(resume_dir)
    #jds = load_documents(jd_dir)
     # Pass resumes & JDs into RAG pipeline
    logging.info(f"Parsing Resume: {resume_file}")
    resume_text = extract_text(str(resume_file))
    
    logging.info(f"Parsing the Job description: {jd_file}")
    jd_text = extract_text(str(jd_file))
    
    if preview:
        print("\n=== Resume Preview ====\n")
        print(resume_text[:500].replace("\n", " "))
        print("\n=== JD Preview ===\n")
        print(jd_text[:500].replace("\n", " "))
        
    # Build the RAG index for resume-JD pair
    vectorstore = build_rag_index(resume_text, jd_text)
    
    # Test RAG retrieval
    logging.info(f"Retrieving top chunks for query: {query}")
    chunks = retrieve_chunks(query, vectorstore)
    print("\n=== Retrieved Chunks ===")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c}\n")
    print(f"Total retrieved chunks: {len(chunks)}")
    
    # Generate and print personalized question
    # Make use of LLM
        
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Prototype: generate a personalized question from one resume+JD"
    )
    p.add_argument("resume_file", type=validate_file,
                   help="Path to resume (.pdf/.txt/.md)")
    p.add_argument("jd_file", type=validate_file,
                   help="Path to job description (.pdf/.txt/.md)")
    p.add_argument("query",
                   type=str,
                   help="Query string to retrieve relevant chunks for testing RAG")
    p.add_argument("--no-preview", action="store_false", dest="preview",
                   help="Skip printing previews")
    args = p.parse_args()

    main(args.resume_file, args.jd_file, args.query, args.preview)
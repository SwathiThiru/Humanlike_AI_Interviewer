#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from data_parser import extract_text

SUPPORTED_EXTS = {".pdf",".txt",".md"}

def find_files(directory: Path):
    """
    Return all files in 'directory' with supported extensions
    """
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

def load_documents(dir_path:Path):
    """
    Extract text from every supported file under dir_path
    
    """
    docs = {}
    for file_path in find_files(dir_path):
        logging.info(f"Parsing {file_path.name}")
        text = extract_text(str(file_path))
        docs[file_path.stem] = text
    return docs

def preview_docs(docs: dict, preview_chars: int = 500):
    """Print the first `preview_chars` of each document."""
    for name, text in docs.items():
        separator = "-" * 5 + f" {name} " + "-" * 5
        print(separator)
        print(text[:preview_chars].replace("\n", " "))
        print()


def main(resume_dir: Path, jd_dir: Path, preview: bool):
    # 1) Load and parse
    resumes = load_documents(resume_dir)
    jds = load_documents(jd_dir)
    
    # 2) Pass resumes & JDs into RAG pipeline
    
    # 3) Quick console preview
    if preview:
        print("\n=== Resumes ===\n")
        preview_docs(resumes)
        print("\n=== Job Descriptions ===\n")
        preview_docs(jds)
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s     %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Parse all resumes and job descriptions in two folders."
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=Path("resume"),
        help="Folder containing resume files"
    )
    parser.add_argument(
        "--jd-dir",
        type=Path,
        default=Path("job_description"),
        help="Folder containing job description files"
    )
    parser.add_argument(
        "--no-preview",
        action="store_false",
        dest="preview",
        help="Skip printing text previews"
    )
    args = parser.parse_args()

    main(args.resume_dir, args.jd_dir, args.preview)
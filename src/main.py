import os
import time

from data_parser import extract_text_from_pdf


def main():
    # Need to configure the inputs here when we have an web interface/application
    # later we can pass the filename directly instead of the path

    interview_duration = 15
    resume_name = "PBMISHRAResume.pdf"
    jd_name = "AI_Software_Engineer_Contentful.pdf"
    resume_path = os.path.join('..', 'resume', resume_name)
    jd_path = os.path.join('..', 'job_description', jd_name)

    # Section 1 : Parse the Documents
    # lets create a modular project

    # Extract the text content from the PDF file
    resume_text = extract_text_from_pdf(resume_path)
    jd_text = extract_text_from_pdf(jd_path)

    # Section 2 : Implement RAG

    # Section 3 : LLM generates question

    # Section 4 : And so on...


if __name__ == "__main__":
    main()
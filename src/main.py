import os
import time

from data_parser import load_and_clean_pdf, chunk_text
from embeddings import VectorStore
from retriever import build_index
from llm import generate_question
from tts import text_to_speech
from stt import transcribe

def main():
    # Need to configure the inputs here when we have an web interface/application
    # later we can pass the filename directly instead of the path

    interview_duration = 15
    resume_name = "PBMISHRAResume.pdf"
    jd_name = "AI_Software_Engineer_Contentful.pdf"
    resume_path = os.path.join('..', 'resume', resume_name)
    jd_path = os.path.join('..', 'job_description', jd_name)

    # Section 1 : Parse the Documents
    resume_text = load_and_clean_pdf(resume_path)
    jd_text = load_and_clean_pdf(jd_path)

    # Section 2 : create chunks
    resume_chunks = chunk_text(resume_text)
    jd_chunks = chunk_text(jd_text)

    # Section 3 : Build vector store
    store = VectorStore()
    store.add(resume_chunks + jd_chunks)
    build_index(resume_chunks, jd_chunks)

    combined = resume_text + jd_text

    # Section 4 : start with interview
    interview_duration = interview_duration * 60
    start = time.time()
    qa_pairs = []

    interview_questions = generate_question("initial", combined)

    while True:
        elapsed = time.time() - start
        if elapsed > interview_duration:
            print("Time’s up")
            break

        # Section 5 : ask questions
        question_audio = text_to_speech(interview_questions)



if __name__ == "__main__":
    main()
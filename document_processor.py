import time

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def extract_skills(resume_text: str, llm, on_retry=None) -> tuple[str, str]:
    """Extract skills summary and candidate name from resume text in one LLM call.

    Returns (skills_summary, candidate_name) so the caller doesn't need a
    second Gemini call just to get the name.

    Caps input at 3000 characters — enough for a full resume without burning
    unnecessary tokens.
    """
    # trim before sending — a resume rarely needs more than 3000 chars to parse
    trimmed = resume_text[:3000]

    prompt = f"""You are a resume parser. From the resume below extract:

NAME: <candidate's full name on one line>

SKILLS:
- <bullet list of every technical skill, language, tool>

PROJECTS:
- <name>: <what it does> | Tech: <technologies>

EXPERIENCE:
- <title> at <company> (<duration>): <key responsibilities>

EDUCATION:
- <degree> in <field>, <institution>, <year>

RESUME:
{trimmed}"""

    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            content = response.content

            # parse the name out so we don't need a second API call
            candidate_name = ""
            for line in content.splitlines():
                if line.upper().startswith("NAME:"):
                    candidate_name = line.split(":", 1)[1].strip()
                    break

            return content, candidate_name
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 15 * (attempt + 1)
                if on_retry:
                    on_retry(wait)
                else:
                    time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini quota exceeded. Please wait a minute and try again.")


if __name__ == "__main__":
    pdf_path = "testfiles/resume.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters")

    chunks = chunk_text(raw_text)
    print(f"Created {len(chunks)} chunks")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    skills_summary = extract_skills(raw_text, llm)
    print(skills_summary)

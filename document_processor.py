
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf(pdf_path:str)->str:
    text="" 
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text+=page_text
    return text

def chunk_text(text:str)->list:
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return splitter.split_text(text)

def extract_skills(resume_text:str,llm)->str:
    prompt=f"""
    You are a resume parser.Analyze the resume below and extract:
    1.Technical Skills-every technology,language and tool mentioned.
    2)Projects- for each:name,what it does,technologies used.
    3)Experience- for each role company ,duration,title  and responsibilities.
    4)Education- for each degree:degree name,field of study,institution and graduation year.
    5)Soft skills-every soft skill mentioned or implied.

    Be specific and thorough in your analysis.Provide the output in a structured format, use bullet points.
    RESUME:{resume_text}
    """
    response=llm.invoke(prompt)
    return response.content

if __name__=="__main__":
    pdf_path="testfiles/resume.pdf"
    print("=" * 50)
    print("STEP 1: Extracting text from PDF")
    print("=" * 50)
    raw_text=extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters\n")
    print("First 300 characters:")
    print(raw_text[:300])

    print("\n" + "=" * 50)
    print("STEP 2: Chunking text")
    print("=" * 50)
    chunks=chunk_text(raw_text)
    print(f"Created {len(chunks)} chunks")
    print(f"\nFirst chunk:\n{chunks[0]}")

    print("\n" + "=" * 50)
    print("STEP 3: Extracting skills with Gemini")
    print("=" * 50)
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7)
    skills_summary=extract_skills(raw_text,llm)
    print(f"Extracted Skills Summary:\n{skills_summary}")



    

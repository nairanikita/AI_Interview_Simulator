from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from document_processor import extract_text_from_pdf,chunk_text

load_dotenv()
def build_vector_store(chunks:list,source:str="resume")->Chroma:
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    docs=[
        Document(
            page_content=chunk,
            metadata={"source":source,"chunk_index":i}
        )
        for i ,chunk in enumerate(chunks)
    ]
    vectorestore=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"

    )
    return vectorestore

def load_vectore_store()->Chroma:
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
def retreive_relevant_chunks(query:str,vectorstore:Chroma,k:int=3)->list:
    results=vectorstore.similarity_search(query,k=k)
    return [doc.page_content for doc in results]

if __name__=="__main__":
    pdf_path="testfiles/resume.pdf"
    print("=" * 50)
    print("STEP 1: Loading and chunking resume")
    print("=" * 50)
    raw_text=extract_text_from_pdf(pdf_path)
    chunks=chunk_text(raw_text)
    print(f"Got {len(chunks)} chunks from resume")

    print("\n" + "=" * 50)
    print("STEP 2: Building vector store")
    print("=" * 50)
    vectorstore=build_vector_store(chunks, source="resume")
    print("ChromaDB built and saved to ./chroma_db/")

    print("\n" + "=" * 50)
    print("STEP 3: Testing retrieval")
    print("=" * 50)
    test_queries=[
        "What machine learning experience does the candidate have?",
        "What cloud platforms has the candidate worked with?",
        "Tell me about the candidate projects"
    ]
    for query in test_queries:
        results=retreive_relevant_chunks(query,vectorstore,k=2)
        for i,chunks in enumerate(results):
            print(f"Result {i+1} for query:{query} \n {chunks}\n")





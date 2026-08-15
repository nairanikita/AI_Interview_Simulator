from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

from document_processor import extract_text_from_pdf, chunk_text

load_dotenv()


def build_vector_store(chunks: list, source: str = "resume", api_key: str = None) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    return vectorstore


def retreive_relevant_chunks(query: str, vectorstore: Chroma, k: int = 3) -> list:
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    pdf_path = "testfiles/resume.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    print(f"Got {len(chunks)} chunks")

    store = build_vector_store(chunks, source="resume")
    results = retreive_relevant_chunks("machine learning experience", store, k=2)
    for i, r in enumerate(results):
        print(f"\nResult {i + 1}:\n{r}")

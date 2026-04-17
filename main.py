import os
import chromadb
from PyPDF2 import PdfReader
from openai import OpenAI


import os
print("Python is looking in:", os.getcwd())

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="insurance_docs",
    metadata={"hnsw:space": "cosine"}
)




# OpenAI client
openai_client = OpenAI()

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [item.embedding for item in response.data]

def ingest_pdfs(pdf_folder="pdfs"):
    doc_id = 0

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            print(f"Ingesting {path}")

            text = load_pdf_text(path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)

            ids = [f"doc_{doc_id+i}" for i in range(len(chunks))]
            doc_id += len(chunks)

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": filename}] * len(chunks)
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_pdfs()
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from tavily import TavilyClient
import os
from typing import Optional
from prompts import AGENDA_PROMPT, PRODUCT_PROMPT

# Initialize FastAPI
app = FastAPI()

# Load persistent ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("insurance_docs")

# OpenAI client
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class QueryRequest(BaseModel):
    question: str
    n_results: int = 2
    session_id: Optional[str] = None




# 1. Connect to your existing database
client = chromadb.PersistentClient(path="chroma_db")


@app.post("/agenda/generate")
def generate_agenda(request: QueryRequest):
    # 1. Search ChromaDB for relevant past notes
    results = collection.query(
        query_texts=[request.question],
        n_results=3
    )


    pdf_chunks = results["documents"][0]

    print(f"===================== {pdf_chunks}")
    pdf_sources = results["metadatas"][0]

    # Step 3 — Internet search using Tavily
    web_results = tavily.search(
        query=request.question,
        max_results=5
    )

    # Build web context
    web_context = ""
    for i, item in enumerate(web_results["results"]):
        web_context += (
            f"WEB RESULT {i+1}\n"
            f"URL: {item['url']}\n"
            f"CONTENT: {item['content']}\n\n"
        )

    # Build PDF context
    pdf_context = ""
    for i, chunk in enumerate(pdf_chunks):
        pdf_context += (
            f"PDF RESULT {i+1}\n"
            f"FILE: {pdf_sources[i]['source']}\n"
            f"CONTENT: {chunk}\n\n"
        )

    print(pdf_context)

    # Step 4 — Combined context
    full_context = pdf_context + "\n" + web_context


    # Step 5 — Stronger prompt for synthesis
    prompt = AGENDA_PROMPT.format() + "\n" + full_context

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("Answer is ready")
    answer = response.choices[0].message.content

    return {
        "question": request.question,
        "answer": answer,
        "pdf_sources": pdf_sources,
        "web_sources": web_results["results"]
    }



@app.post("/products/suggest")
async def suggest_products(request: QueryRequest):
    # 1. Search ChromaDB for relevant past notes
    results = collection.query(
        query_texts=[request.question],
        n_results=3
    )

    pdf_chunks = results["documents"][0]

    print(f"===================== {pdf_chunks}")
    pdf_sources = results["metadatas"][0]

    # Step 3 — Internet search using Tavily
    web_results = tavily.search(
        query=request.question,
        max_results=5
    )

    # Build web context
    web_context = ""
    for i, item in enumerate(web_results["results"]):
        web_context += (
            f"WEB RESULT {i + 1}\n"
            f"URL: {item['url']}\n"
            f"CONTENT: {item['content']}\n\n"
        )

    # Build PDF context
    pdf_context = ""
    for i, chunk in enumerate(pdf_chunks):
        pdf_context += (
            f"PDF RESULT {i + 1}\n"
            f"FILE: {pdf_sources[i]['source']}\n"
            f"CONTENT: {chunk}\n\n"
        )

    print(pdf_context)

    # Step 4 — Combined context
    full_context = pdf_context + "\n" + web_context

    # Step 5 — Stronger prompt for synthesis
    prompt = PRODUCT_PROMPT.format() + "\n" + full_context

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print("Answer is ready")
    answer = response.choices[0].message.content

    return {
        "question": request.question,
        "answer": answer,
        "pdf_sources": pdf_sources,
        "web_sources": web_results["results"]
    }
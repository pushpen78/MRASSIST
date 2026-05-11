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


@app.post("/agenda/generate")
def generate_agenda(request: QueryRequest):
    # 1. Search ChromaDB for relevant past notes
    results = collection.query(
        query_texts=[request.question],
        n_results=3
    )
    local_context = "\n".join(results['documents'][0])

    # 2. Format and Send to LLM
    formatted_prompt = AGENDA_PROMPT.format(
        full_context=f"PAST NOTES: {local_context}",
        question=request.question
    )

    # response = llm.invoke(formatted_prompt)
    return {"agenda": "Your generated result"}


@app.post("/products/suggest")
async def suggest_products(request: QueryRequest):
    # 1. Search internal inventory (ChromaDB)
    db_results = collection.query(query_texts=[request.question], n_results=2)
    internal_data = "\n".join(db_results['documents'][0])

    # 2. Get real-time data/reviews from Web (Tavily)
    web_context = tavily.get_search_context(query=request.question)

    # 3. Combine contexts
    full_context = f"INTERNAL INVENTORY: {internal_data}\n\nWEB RESEARCH: {web_context}"

    # 4. Format Prompt
    formatted_prompt = PRODUCT_PROMPT.format(
        full_context=full_context,
        question=request.question
    )

    # response = llm.invoke(formatted_prompt)
    return {"suggestions": "Your 3 healthcare products"}
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from tavily import TavilyClient
import os

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

print("outside the request")


@app.post("/answer")
def answer_question(request: QueryRequest):
    print("Inside the request function")
    # Step 1 — Embed the question using OpenAI (lightweight)
    embedding_response = llm.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = embedding_response.data[0].embedding

    print(f"DEBUG: request.n_results is {request.n_results}")
    print(f"DEBUG: session_id is {request.session_id}")
    # Step 2 — Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results
    )

    pdf_chunks = results["documents"][0]
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



    # Step 4 — Combined context
    full_context = pdf_context + "\n" + web_context

    # Step 5 — Stronger prompt for synthesis
    prompt = f"""
Act as an Elite Medical Sales Representative specializing in Metabolic Health. 
You are conducting a "Scientific Summary" call with an Endocrinologist regarding Product X.


Your job is to get data ONLY from PDF

Knowledge Base & Research:
Product X Ingested Data: Prioritize the Scientific Summary, Phase III trial results (e.g., glycemic control, weight loss, or renal outcomes), and the Prescribing Information (PI) found in the PDFs.
External Context: Use the internet to reference current ADA (American Diabetes Association) or AACE (American Association of Clinical Endocrinology) guidelines to show how Product X fits into the latest standards of care.

HCP Persona (The Endocrinologist):
Priorities: They are data-driven experts who value evidence-based medicine. They care about long-term efficacy, cardiovascular/renal safety profiles, and minimizing patient "therapeutic inertia".
Pain Points: Patient adherence to complex dosing, managing comorbidities (obesity, CKD), and the administrative burden of prior authorizations.

Operational Guidelines:
The "Call Steps" Format:
Opening: A high-impact, clinical hook—not a greeting. (e.g., "Doctor, given the recent focus on cardio-renal protection in Type 2 Diabetes...").
The Scientific "Deep Dive": Present specific data points from the PDFs. Focus on 
-values, 
 sizes, and hazard ratios.
HCP Perspective Integration: Acknowledge their specific challenges, such as "sick day guidance" or hypoglycemia risks.
Closing: A specific "Call to Action"—requesting a follow-up to discuss a specific patient type or providing a sample kit.

Compliance Check:
Include a mandatory "Important Safety Information (ISI)" section highlighting the most common adverse events and contraindications from the PI.

Then produce a clear, natural-language answer.

### CONTEXT START ###
{full_context}
### CONTEXT END ###

### USER QUESTION ###
{request.question}
"""

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
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq

app = FastAPI(title="Document Copilot RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== CONFIG ======================
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

client = chromadb.PersistentClient(path=PERSIST_DIR)
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY environment variable is not set. Please set it before running the server.")

groq_client = Groq(api_key=groq_api_key)

class QueryRequest(BaseModel):
    question: str

# ====================== API ENDPOINTS ======================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query")
async def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    try:
        results = collection.query(
            query_texts=[request.question],
            n_results=6,
            include=["documents", "metadatas"]
        )

        contexts = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_str = "\n\n".join([
            f"Source: {meta.get('source', 'unknown')}\n{doc}"
            for doc, meta in zip(contexts, metadatas)
        ])

        # Single-line prompt to avoid quote issues
        prompt = "Answer the question using ONLY the provided context.\n\nContext:\n" + context_str + "\n\nQuestion: " + request.question + "\n\nAnswer clearly and concisely. If the information is not in the context, say \"I don't have enough information from the documents.\""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )

        answer = completion.choices[0].message.content.strip()
        sources = list({meta.get("source") for meta in metadatas if meta.get("source")})

        return {"answer": answer, "sources": sources}

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        return {"answer": "Sorry, an internal error occurred.", "sources": []}

# ====================== SERVE UI ======================
@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Document Copilot at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

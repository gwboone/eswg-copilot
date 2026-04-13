import os
import uuid

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from pypdf import PdfReader
from docx import Document
import docx2txt
from striprtf.striprtf import rtf_to_text

def load_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"⚠️ Error reading PDF {file_path}: {e}")
        return ""

def load_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"⚠️ Error reading DOCX {file_path}: {e}")
        return ""

def load_doc(file_path: str) -> str:
    try:
        return docx2txt.process(file_path) or ""
    except Exception as e:
        print(f"⚠️ Error reading .doc {file_path}: {e}")
        return ""

def load_rtf(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return rtf_to_text(content) or ""
    except Exception as e:
        print(f"⚠️ Error reading RTF {file_path}: {e}")
        return ""

def load_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Error reading TXT {file_path}: {e}")
        return ""

def get_text_from_file(file_path: str) -> str | None:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return load_pdf(file_path)
    elif ext == ".docx": return load_docx(file_path)
    elif ext == ".doc": return load_doc(file_path)
    elif ext == ".rtf": return load_rtf(file_path)
    elif ext == ".txt": return load_txt(file_path)
    else:
        print(f"ℹ️ Skipping unsupported file: {file_path}")
        return None

def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def ingest(docs_dir: str = "./docs", persist_dir: str = "./chroma_db", collection_name: str = "documents"):
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"✅ Created {docs_dir}. Add your documents and run again.")
        return

    print("🔄 Connecting to ChromaDB...")

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )

    documents = []
    metadatas = []
    ids = []

    print(f"📂 Scanning {docs_dir}...")

    supported = {".pdf", ".docx", ".doc", ".rtf", ".txt"}

    for root, _, files in os.walk(docs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext not in supported:
                continue

            text = get_text_from_file(file_path)
            if not text or not text.strip():
                continue

            chunks = get_text_chunks(text)
            source = os.path.basename(file_path)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                doc_id = f"{source}_{uuid.uuid4().hex[:12]}"
                documents.append(chunk)
                metadatas.append({"source": source, "chunk_index": i})
                ids.append(doc_id)

    if documents:
        print(f"📦 Adding {len(documents)} chunks...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        unique_files = len(set(m["source"] for m in metadatas))
        print(f"✅ Success! Ingested {unique_files} file(s) → {len(documents)} chunks")
    else:
        print("⚠️ No valid text found in documents.")

if __name__ == "__main__":
    print("🚀 Document Copilot Ingestion Started\n")
    ingest()
    print("\n✅ Ingestion completed!")

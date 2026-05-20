import requests
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import io
import re

# --- Download the paper ---
url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
response = requests.get(url)
pdf_bytes = io.BytesIO(response.content)

# --- Extract text from PDF ---
reader = PdfReader(pdf_bytes)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Extracted {len(full_text)} characters from paper")


# --- Clean extracted text ---
# PDF extraction often has broken hyphenation and extra whitespace
def clean_text(text):
    # Fix broken hyphenated words across lines
    text = re.sub(r'-\n', '', text)
    # Collapse multiple spaces but NOT newlines
    text = re.sub(r' +', ' ', text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'\n\d+\n', '\n', text)
    # DO NOT collapse \n to space — we need them for chunking
    return text

full_text = clean_text(full_text)


# --- Sentence-aware chunker with overlap ---
def chunk_text(text, chunk_size=400, overlap=100):
    """
    Single-newline aware chunker for PDF-extracted text.
    Splits on \n lines, merges into chunks of target size.
    """
    # Split on single newlines — this is the paragraph separator in this PDF
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap — keep last N chars
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + line
        else:
            current_chunk += " " + line

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks



print(f"\nFirst 500 chars of cleaned text:")
print(repr(full_text[:500]))  # repr shows \n as literal \n so you can see structure
print(f"\nNewline count: {full_text.count(chr(10))}")
print(f"Double newline count: {full_text.count(chr(10)*2)}")

chunks = chunk_text(full_text)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):  # show first 3, safe if fewer exist
    print(f"Chunk {i}: [{chunk[:100]}]")


# --- Embed chunks ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)

# --- Rebuild ChromaDB from scratch ---
import shutil
import os

db_path = "./attention_paper_db"

# Delete BEFORE opening ChromaDB — must happen first to avoid file locks
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Deleted existing database at {db_path}")
else:
    print("No existing database — creating fresh")

# Now create client on clean empty folder
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("attention_paper")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Stored {len(chunks)} chunks in ChromaDB")
print("Ingestion complete — run chatbot.py to start chatting")
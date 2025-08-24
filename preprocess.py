import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load PDF
pdf_path = "sodapdf-converted.pdf"
try:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    logger.info(f"Loaded PDF: {len(text)} characters")
except FileNotFoundError:
    logger.error(f"PDF not found: {pdf_path}")
    raise

# Split into chunks (small PDF, so use small chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(text)
logger.info(f"Created {len(chunks)} chunks")

# Generate embeddings
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
chunk_embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
logger.info("Generated embeddings")

# Save to JSON
output = {"chunks": chunks, "embeddings": chunk_embeddings}
with open("embeddings.json", "w") as f:
    json.dump(output, f, indent=2)

logger.info(f"Embeddings saved to embeddings.json! {len(chunks)} chunks created.")
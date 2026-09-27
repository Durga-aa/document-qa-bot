import os
import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# SETTINGS
# ==========================================

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "document_collection"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 32


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ==========================================
# 2. CLEAN TEXT
# ==========================================

def clean_text(text):

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces around text
    text = text.strip()

    return text


# ==========================================
# 3. SPLIT TEXT INTO CHUNKS
# ==========================================

def create_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# 4. LOAD PDF DOCUMENTS
# ==========================================

def load_documents():

    all_chunks = []
    all_metadatas = []

    pdf_files = [
        file for file in os.listdir(DATA_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    print(f"\nFound {len(pdf_files)} PDF files.")

    for filename in pdf_files:

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        print(f"\nProcessing: {filename}")

        reader = PdfReader(file_path)

        print(f"Pages: {len(reader.pages)}")

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)

            chunks = create_chunks(text)

            for chunk in chunks:

                all_chunks.append(chunk)

                all_metadatas.append(
                    {
                        "source": filename,
                        "page": page_number
                    }
                )

    return all_chunks, all_metadatas


# ==========================================
# 5. MAIN INDEXING PROCESS
# ==========================================

def main():

    print("\n" + "=" * 60)
    print("          DOCUMENT INDEXING")
    print("=" * 60)

    # Load and chunk documents
    chunks, metadatas = load_documents()

    print("\nTotal chunks created:", len(chunks))

    if len(chunks) == 0:

        print(
            "ERROR: No chunks were created."
        )

        return

    # ======================================
    # GENERATE EMBEDDINGS IN BATCHES
    # ======================================

    print("\nGenerating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True
    )

    embeddings = embeddings.tolist()

    print("Embeddings generated successfully!")


    # ======================================
    # CONNECT TO CHROMADB
    # ======================================

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    # Delete old collection if it exists
    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print("Old collection removed.")

    except Exception:

        print("No existing collection found.")


    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME
    )


    # ======================================
    # STORE CHUNKS + EMBEDDINGS
    # ======================================

    print("\nStoring data in ChromaDB...")

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\nIndexing completed successfully!")

    print(
        "Total chunks stored:",
        collection.count()
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()
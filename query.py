import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="document_collection"
)

# ==========================================
# 2. Load embedding model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# 3. Load FLAN-T5 model
# ==========================================

print("\nLoading answer generation model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

llm = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

print("Answer generation model loaded!")

# ==========================================
# 4. Start application
# ==========================================

print("\n====================================")
print("DOCUMENT Q&A BOT")
print("====================================")

print("Documents available:", collection.count())

question = input("\nAsk your question: ")

# ==========================================
# 5. Create question embedding
# ==========================================

query_embedding = embedding_model.encode(
    question
).tolist()

# ==========================================
# 6. Retrieve relevant chunks
# ==========================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

# ==========================================
# 7. Prepare context
# ==========================================

context_parts = []

for document, metadata in zip(documents, metadatas):

    source = metadata.get("source", "Unknown")
    page = metadata.get("page", "Unknown")

    context_parts.append(
        f"Source: {source}, Page: {page}\n"
        f"{document[:1200]}"
    )

context = "\n\n".join(context_parts)

# Keep context within model limit
context = context[:4000]

# ==========================================
# 8. Create prompt
# ==========================================

prompt = f"""
Answer the question using only the information provided in the context.

If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Give a short and clear answer.

Question:
{question}

Context:
{context}

Answer:
"""

# ==========================================
# 9. Generate answer
# ==========================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

outputs = llm.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# ==========================================
# 10. Display answer
# ==========================================

print("\n====================================")
print("ANSWER")
print("====================================")

print("\n" + answer.strip())

# ==========================================
# 11. Display citations
# ==========================================

print("\n====================================")
print("SOURCES")
print("====================================")

seen = set()

for metadata in metadatas:

    source = metadata.get("source", "Unknown")
    page = metadata.get("page", "Unknown")

    citation = (source, page)

    if citation not in seen:

        print(f"- {source}, Page {page}")

        seen.add(citation)

print("\n====================================")
print("DONE")
print("====================================")
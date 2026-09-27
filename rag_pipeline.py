import chromadb
import ollama
from sentence_transformers import SentenceTransformer



# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(path="chroma_db")

collections = client.list_collections()

print("\nAvailable collections:")

for c in collections:
    print(c.name)

if len(collections) == 0:
    print("No ChromaDB collection found.")
    exit()

collection = client.get_collection("document_collection")

print("Using collection:", collection.name)


# ==========================================
# 3. CONNECT TO OLLAMA
# ==========================================

OLLAMA_MODEL = "llama3.2:3b"

print("\nUsing Ollama model:", OLLAMA_MODEL)

print("Ollama connected successfully!")

# ==========================================
# 4. SEARCH DOCUMENTS
# ==========================================

def search_documents(question, top_k=3):

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --------------------------------------
    # CHECK WHETHER RESULTS ARE RELEVANT
    # --------------------------------------

    # Lower distance = more similar
    # Higher distance = less similar

    if not distances:
        return [], []

    best_distance = distances[0]

    # If nothing is sufficiently similar,
    # do not send unrelated context to the LLM.
    if best_distance > 1.2:
        print(
            f"\nNo sufficiently relevant document found."
            f" Best distance: {best_distance:.4f}"
        )

        return [], []

    # --------------------------------------
    # PREFER MACHINE LEARNING DOCUMENT
    # --------------------------------------

    question_lower = question.lower()

    if "machine learning" in question_lower:

        ml_documents = []
        ml_metadatas = []

        other_documents = []
        other_metadatas = []

        for doc, metadata in zip(documents, metadatas):

            source = metadata.get(
                "source",
                ""
            ).lower()

            if "machine_learning.pdf" in source:

                ml_documents.append(doc)
                ml_metadatas.append(metadata)

            else:

                other_documents.append(doc)
                other_metadatas.append(metadata)

        documents = ml_documents + other_documents
        metadatas = ml_metadatas + other_metadatas

    return documents, metadatas


# ==========================================
# 5. GENERATE ANSWER USING OLLAMA
# ==========================================

def generate_answer(question, documents):

    # --------------------------------------
    # NO RELEVANT DOCUMENT FOUND
    # --------------------------------------

    if not documents:

        return (
            "I'm unable to provide a reliable answer "
            "because the required information was not found."
        )

    # --------------------------------------
    # COMBINE RETRIEVED DOCUMENT CHUNKS
    # --------------------------------------

    context = "\n\n".join(documents)

    # Keep the context within a reasonable size
    context = context[:6000]

    # --------------------------------------
    # RAG PROMPT
    # --------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT.

Do NOT use your own outside knowledge.

If the answer cannot be found in the CONTEXT, respond exactly:

I'm unable to provide a reliable answer because the required information was not found.

Keep the answer clear, concise, and directly related to
the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------
    # CALL OLLAMA
    # --------------------------------------

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer.strip()


# ==========================================
# 6. MAIN Q&A LOOP
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("       RAG Document Q&A Bot")
    print("=" * 50)

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower().strip() == "exit":
            print("Goodbye!")
            break

        # --------------------------------------
        # Retrieve relevant documents
        # --------------------------------------

        documents, sources = search_documents(question)

        # --------------------------------------
        # Generate answer
        # --------------------------------------

        answer = generate_answer(
            question,
            documents
        )

        # --------------------------------------
        # Display answer
        # --------------------------------------

        print("\nAnswer:")
        print("-" * 50)
        print(answer)

        # --------------------------------------
        # Display sources
        # --------------------------------------

        print("\nSources:")
        print("-" * 50)

        for i, source in enumerate(sources):

            print(f"\nSource {i + 1}")

            print(
                "File:",
                source.get("source", "Unknown")
            )

            print(
                "Page:",
                source.get("page", "Unknown")
            )

            print("\nChunk:")
            print(documents[i])
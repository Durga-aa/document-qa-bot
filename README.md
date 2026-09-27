# 📚 Document Q&A Bot

A basic Retrieval-Augmented Generation (RAG) based Document Question Answering system.

The application allows users to ask questions about a collection of PDF documents. It retrieves relevant document chunks from a persistent ChromaDB vector database and uses the Ollama LLM to generate answers based only on the retrieved document content.

## 🚀 Features

- PDF document ingestion
- Text extraction from PDF files
- Text chunking with overlap
- Source metadata including filename and page number
- Sentence Transformer embeddings
- Persistent ChromaDB vector database
- Similarity-based document retrieval
- Configurable Top-K retrieval
- Ollama LLM for answer generation
- Answers restricted to retrieved document context
- Source file and page number display
- Source chunk display
- Handles questions that are not present in the documents
- Command-line interface (CLI)
- Streamlit web interface

## 🏗️ Architecture

User Question
       ↓
Question Embedding
       ↓
ChromaDB Similarity Search
       ↓
Top-K Relevant Document Chunks
       ↓
Context Construction
       ↓
Ollama LLM
       ↓
Answer
       ↓
Source File + Page + Chunk

## 📂 Project Structure

document-qa-bot/
│
├── data/
│   ├── python.pdf
│   ├── machine_learning.pdf
│   ├── artificial_intelligence.pdf
│   ├── cloud_computing.pdf
│   └── databases.pdf
│
├── chroma_db/
│
├── rag_pipeline.py
├── app.py
├── README.md
└── requirements.txt

## 🔧 Technologies Used

- Python 3.11+
- pypdf
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2 3B
- Streamlit

## 🧩 Chunking Strategy

The PDF text is divided into smaller chunks before creating embeddings.

Chunking helps the retrieval system find the specific portions of documents that are relevant to a user's question.

Each chunk is stored with metadata such as:

- Source filename
- Page number

This metadata is later displayed as the source citation.

## 🔎 Retrieval

The system converts the user's question into an embedding using the same embedding model used for the document chunks.

ChromaDB performs similarity search and retrieves the most relevant chunks.

The system currently retrieves the top 3 relevant chunks.

A similarity threshold is also used to prevent unrelated questions from being answered using irrelevant documents.

## 🤖 LLM

The application uses:

Ollama  
Model: `llama3.2:3b`

The LLM receives only the retrieved document context.

The prompt instructs the model not to use outside knowledge.

If relevant information cannot be found in the retrieved documents, the application returns:

"I couldn't find this information in the provided documents."

## 🗄️ Vector Database

ChromaDB is used as the vector database.

The document embeddings and metadata are stored persistently so that the indexed documents can be reused without rebuilding the entire database every time.

Collection name:

`document_collection`

## ▶️ Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv
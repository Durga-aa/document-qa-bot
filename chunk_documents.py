from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder containing the PDF files
DATA_DIR = Path("data")

# Find all PDF files
pdf_files = list(DATA_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Store all chunks
all_chunks = []

# Read and split each PDF
for pdf_file in pdf_files:
    print(f"Processing: {pdf_file.name}")

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Split page text into chunks
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": pdf_file.name,
                "page": page_number
            })

    print(f"  Pages: {len(reader.pages)}")
    print("-" * 50)

# Display results
print(f"\nTotal chunks created: {len(all_chunks)}")

print("\nFirst 5 chunks:\n")

for i, chunk in enumerate(all_chunks[:5], start=1):
    print(f"Chunk {i}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:300]}")
    print("-" * 50)
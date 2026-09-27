from pathlib import Path
from pypdf import PdfReader

# Folder containing the PDF files
DATA_DIR = Path("data")

# Find all PDF files
pdf_files = list(DATA_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

# Read each PDF
for pdf_file in pdf_files:
    print(f"Reading: {pdf_file.name}")

    reader = PdfReader(pdf_file)

    print(f"Number of pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        print(
            f"  Page {page_number}: "
            f"{len(text)} characters"
        )

    print("-" * 50)

print("All PDF files loaded successfully!")
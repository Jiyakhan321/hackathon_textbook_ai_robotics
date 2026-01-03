import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from rag.ingest import ingest_documents

load_dotenv()

async def index_book_content():
    """
    Index the book content from the Docusaurus docs folder into Qdrant
    """
    print("Starting book content indexing...")
    
    # Get the path to the Docusaurus docs
    docs_path = Path("../../my-website/docs")  # Adjust path as needed
    
    if not docs_path.exists():
        print(f"Docs path does not exist: {docs_path}")
        return
    
    documents = []
    
    # Read all markdown files in the docs directory
    for md_file in docs_path.rglob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Create document object
                doc = {
                    "id": str(md_file.relative_to(docs_path)),
                    "title": md_file.name,
                    "content": content,
                    "source": str(md_file)
                }
                
                documents.append(doc)
                print(f"Added document: {doc['title']}")
                
        except Exception as e:
            print(f"Error reading {md_file}: {str(e)}")
    
    if documents:
        print(f"Found {len(documents)} documents. Starting ingestion...")
        result = await ingest_documents(documents)
        print(f"Ingestion completed: {result}")
    else:
        print("No documents found to ingest")

if __name__ == "__main__":
    asyncio.run(index_book_content())
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
from PyPDF2 import PdfReader

PDF_CONTRACT_PATH = "data/Sample_Contract_GEP_ABC.pdf"
PDF_INVOICE_PATH = "data/Sample_Invoice_GEP_ABC.pdf"


embeddings = AzureOpenAIEmbeddings(
       azure_endpoint="https://openaiqc.gep.com/techathon/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="text-embedding-3-large", 
    api_version="2023-03-15-preview" 
)

db_location = "./chroma_db"
add_documents = not os.path.exists(db_location)

def extract_pdf_pages(pdf_path):
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append((i + 1, text.strip()))
    except FileNotFoundError:
        print(f"[WARN] PDF not found: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {pdf_path}: {e}")
    return pages

if add_documents:
    documents = []

    contract_pages = extract_pdf_pages(PDF_CONTRACT_PATH)
    if contract_pages:
        for page_no, text in contract_pages:
            if not text:
                continue
            content = f"Source: Sample Contract\nPage: {page_no}\n\n{text}"
            metadata = {
                "source": os.path.basename(PDF_CONTRACT_PATH),
                "doc_type": "contract",
                "page": page_no,
            }
            documents.append(Document(page_content=content, metadata=metadata))
    else:
        print(f"[INFO] No pages extracted from {PDF_CONTRACT_PATH}")

    invoice_pages = extract_pdf_pages(PDF_INVOICE_PATH)
    if invoice_pages:
        for page_no, text in invoice_pages:
            if not text:
                continue
            content = f"Source: Sample Invoice\nPage: {page_no}\n\n{text}"
            metadata = {
                "source": os.path.basename(PDF_INVOICE_PATH),
                "doc_type": "invoice",
                "page": page_no,
            }
            documents.append(Document(page_content=content, metadata=metadata))
    else:
        print(f"[INFO] No pages extracted from {PDF_INVOICE_PATH}")

    if documents:
        print(f"[INFO] Creating Chroma DB at {db_location} with {len(documents)} documents...")
        vector_store = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=db_location
        )
        vector_store.persist()
        print("[INFO] Persisted Chroma DB.")
    else:
        print("[WARN] No documents to add. Chroma DB not created.")

vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=db_location
    )


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

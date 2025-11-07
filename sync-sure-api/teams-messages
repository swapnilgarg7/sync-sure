
import os
import json
import tempfile
import uvicorn
from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    BotFrameworkAdapter,
    TurnContext
)
from botbuilder.schema import Activity

load_dotenv()

# Initialize model (same as your snippet)
model = AzureChatOpenAI(
    azure_endpoint="https://openaiqc.gep.com/techathon/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name="gpt-4.1", 
    api_version="2023-03-15-preview" 
)

template = """
You are GEP SyncSure AI, an expert financial compliance assistant developed by GEP.
Your job is to verify supplier invoices against contract terms and detect potential non-compliance, overbilling, or data mismatches.

... (KEEP the same template text you originally had) ...

### CONTRACT DATA
{contract}

### INVOICE DATA
{invoice}
"""
# Use your original template content — truncated here for brevity in this file listing.
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def extract_pdf_text(pdf_path):
    full_text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text + "\n\n"
    except Exception as e:
        print("[ERROR] extract_pdf_text:", e)
    return full_text.strip()

def extract_docx_text(docx_path):
    full_text = ""
    try:
        document = docx.Document(docx_path)
        for para in document.paragraphs:
            full_text += para.text + "\n"
    except Exception as e:
        print("[ERROR] extract_docx_text:", e)
    return full_text.strip()

def load_document_text(file_path, original_filename):
    if original_filename.lower().endswith(".pdf"):
        return extract_pdf_text(file_path)
    elif original_filename.lower().endswith(".docx"):
        return extract_docx_text(file_path)
    else:
        return None

def analyze_document_files(contract_path, contract_filename, invoice_path, invoice_filename):
    """
    Loads files, calls the LLM pipeline, and returns a Python dict (the parsed JSON).
    """
    contract_text = load_document_text(contract_path, contract_filename)
    invoice_text = load_document_text(invoice_path, invoice_filename)

    if not contract_text or not invoice_text:
        raise Exception("Failed to extract text from one or both files.")

    # Invoke the chain synchronously (your snippet used chain.invoke)
    print("[INFO] Invoking model...")
    result = chain.invoke({
        "contract": contract_text,
        "invoice": invoice_text
    })

    # result.content expected to be a string of JSON as per your template output format
    try:
        parsed = json.loads(result.content)
        return parsed
    except Exception as e:
        # If the LLM didn't return valid JSON, return a structured error object
        return {
            "error": "Failed to parse LLM output as JSON",
            "raw_output": getattr(result, "content", str(result)),
            "exception": str(e)
        }


MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID", "")

# Bot adapter
adapter_settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID)
adapter = BotFrameworkAdapter(adapter_settings)

app = FastAPI(title="GEP SyncSure API", version="1.0.0")

# --- Bot message endpoint for Teams ---
@app.post("/messages")
async def messages(req: Request):
    """
    Endpoint that Bot Framework / Teams will POST Activity objects to.
    """
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    async def aux_handler(turn_context: TurnContext):
        # Minimal handler: respond to certain commands
        if turn_context.activity.type == "message":
            text = (turn_context.activity.text or "").strip().lower()
            if text == "analyze" or "analyze" in text:
                await turn_context.send_activity(
                    "To analyze invoices, please POST the contract and invoice files (multipart/form-data) to `POST /analyze-invoice` on this service. "
                    "Fields: `contract` and `invoice`. The endpoint returns a JSON compliance report."
                )
                await turn_context.send_activity("Example: Use curl or Postman to upload both files.")
            else:
                # basic echo for now
                await turn_context.send_activity(f"Echo: {turn_context.activity.text}")

        else:
            await turn_context.send_activity("Received activity of type: " + turn_context.activity.type)

    try:
        # adapter.process_activity returns a result but for FastAPI we simply return 200
        await adapter.process_activity(activity, auth_header, aux_handler)
        return Response(status_code=200)
    except Exception as e:
        print("[ERROR] Error processing activity:", e)
        return Response(status_code=500, content=str(e))


# --- Keep the file-upload analysis endpoint (reuse your original logic) ---
@app.post("/analyze-invoice")
async def analyze_invoice(
    contract: UploadFile = File(..., description="Contract file (.pdf or .docx)"),
    invoice: UploadFile = File(..., description="Invoice file (.pdf or .docx)")
):
    """
    Accepts multi-part upload (contract + invoice), runs the SyncSure analyzer,
    and returns a JSON compliance report.
    """
    contract_path = None
    invoice_path = None
    try:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(contract.filename)[1]) as tmpc:
            tmpc.write(await contract.read())
            contract_path = tmpc.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(invoice.filename)[1]) as tmpi:
            tmpi.write(await invoice.read())
            invoice_path = tmpi.name

        # Call the analyzer function (wrapping your LangChain model)
        report = analyze_document_files(contract_path, contract.filename, invoice_path, invoice.filename)
        # report expected to be a JSON-serializable dict
        return report

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR] analyze_invoice:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (contract_path, invoice_path):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    print("Starting GEP SyncSure bot server at http://127.0.0.1:3978")
    uvicorn.run(app, host="0.0.0.0", port=3978)

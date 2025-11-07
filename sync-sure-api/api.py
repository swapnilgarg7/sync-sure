from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
# from vector import retriever
from PyPDF2 import PdfReader
import docx
import tempfile
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from document_parser import load_document_text

load_dotenv()

app = FastAPI(
    title="GEP SyncSure API",
    description="API for comparing supplier invoices against contract terms.",
    version="1.0.0"
)

model = AzureChatOpenAI(
    azure_endpoint="https://openaiqc.gep.com/techathon/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name="gpt-4.1", 
    api_version="2023-03-15-preview" 
)

template = """
You are GEP SyncSure AI, an expert financial compliance assistant developed by GEP.
Your job is to verify supplier invoices against contract terms and detect potential non-compliance, overbilling, or data mismatches.

### Context
You are given two inputs:
1. CONTRACT DATA — contains commercial terms, rates, validity, and payment clauses.
2. INVOICE DATA — contains billed line items, quantities, prices, and totals.

### Task
Perform a detailed comparison between the contract and the invoice and identify:
- Any **price mismatches**
- Any **quantity or unit discrepancies**
- Any **billing outside contract validity period**
- Any **missing or incorrect taxes/discounts**
- Any **payment terms deviation**

Then:
1. **List all detected non-compliances**, with a short explanation.
2. **Provide a compliance score (0–100%)** for your assessment.
3. **Recommend an action** for each issue:
   - Auto-approve 
   - Flag for review 
   - Reject 
   - Request supplier clarification 

### Output Format
Respond strictly in JSON:
{{
  "summary": "Overall compliance status (Compliant / Non-Compliant)",
  "compliance-score": 0–100,
  "issues": [
    {{
      "type": "Price Mismatch / Quantity Error / Term Violation / Discount Error / Other",
      "description": "What’s wrong",
      "contract_reference": "Relevant clause or term",
      "invoice_reference": "Item or field name",
      "suggested_action": "Auto-approve / Flag / Reject / Clarify",
      "severity": "Low / Medium / High"
    }}
  ],
  "recommendation": "Overall next step (approve, flag, or reject)",
  "notes": "Optional business insight or caution"
}}

### Guidelines
- Always use professional and concise language.
- Use logical reasoning to cross-reference data fields.
- Never hallucinate contract clauses not present in the input.
- If data is incomplete, state assumptions clearly.

Now begin the analysis using the provided CONTRACT DATA and INVOICE DATA.

### CONTRACT DATA
{contract}

### INVOICE DATA
{invoice}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def save_upload_file_temp(upload_file: UploadFile) -> str:
    """Saves UploadFile to a temporary file and returns the path."""
    try:
        # Create a named temporary file,
        # 'delete=False' means the file is not deleted when closed
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as temp_file:
            temp_file.write(upload_file.file.read())
            return temp_file.name
    except Exception as e:
        print(f"[ERROR] Could not save temp file: {e}")
        return None
    finally:
        upload_file.file.close()


# --- API Endpoint ---
@app.post("/analyze-invoice")
async def analyze_invoice(
    contract: UploadFile = File(..., description="The contract file (.pdf or .docx)"),
    invoice: UploadFile = File(..., description="The invoice file (.pdf or .docx)")
):
    """
    Analyzes an invoice against a contract and returns a JSON compliance report.
    """
    contract_path = None
    invoice_path = None

    try:
        # Save uploaded files to temporary paths
        contract_path = save_upload_file_temp(contract)
        invoice_path = save_upload_file_temp(invoice)

        if not contract_path or not invoice_path:
            raise HTTPException(status_code=500, detail="Failed to save uploaded files.")

        # Load text from the temporary files
        contract_full_text = load_document_text(contract_path, contract.filename)
        invoice_full_text = load_document_text(invoice_path, invoice.filename)

        if not contract_full_text or not invoice_full_text:
            raise HTTPException(status_code=415, detail="Failed to read one or both documents. Ensure they are valid .pdf or .docx files.")

        # Run the LangChain chain
        print("[INFO] Both documents loaded. Invoking chain...")
        result = chain.invoke({
            "contract": contract_full_text,
            "invoice": invoice_full_text
        })
        
        # The model returns a string, which we need to parse into JSON
        try:
            json_response = json.loads(result.content)
            return json_response
        except json.JSONDecodeError:
            print("[ERROR] LLM output was not valid JSON.")
            # If the LLM fails to return JSON, return its raw response in a standard error object
            raise HTTPException(status_code=500, detail={
                "error": "Failed to parse LLM response as JSON.",
                "raw_output": result.content
            })

    except Exception as e:
        print(f"[ERROR] An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    
    finally:
        # Clean up the temporary files
        if contract_path and os.path.exists(contract_path):
            os.unlink(contract_path)
        if invoice_path and os.path.exists(invoice_path):
            os.unlink(invoice_path)


# --- Run the App (for local testing) ---
if __name__ == "__main__":
    print("Starting GEP SyncSure AI API server at http://127.0.0.1:8000")
    print("View interactive API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

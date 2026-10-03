from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import pypdf
import tempfile
import uvicorn
from pydantic import BaseModel
import logging

# Move langserve import to the top.
from langserve import add_routes

from chain import create_summarization_chain

class SummarizeBatchRequest(BaseModel):
    text: str

# Create FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="An API for summarizing text using Google's Gemini model",
    version="1.0.0",
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>FastAPI Home</title>
        </head>
        <body>
            <h1>Welcome to LangServe API!</h1>
            <p>Click below to summarise your text in Langserve playground:</p>
            <a href="/summarize/playground" style="font-size: 18px; color: blue;">Langserve Playground</a>
            <h1>Test FastAPI!</h1>
            <p>Click below to access the API documentation and test the API:</p>
            <a href="/docs" style="font-size: 18px; color: blue;">Go to Swagger UI</a>
        </body>
    </html>
    """

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm-summarizer")

# Create the summarization chain
summarization_chain = create_summarization_chain()

# Add LangServe route with explicit input/output typing
add_routes(
    app,
    summarization_chain,
    path="/summarize",
    input_type=SummarizeBatchRequest,
)

@app.post("/summarize-pdf/", description="Upload a PDF file to be summarized")
async def summarize_pdf(file: UploadFile = File(...)):
    # Check if the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
        
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Read the uploaded file content
            content = await file.read()
            # Write the content to the temporary file
            temp_file.write(content)
            temp_file.flush()
                        
            # Extract text from the PDF
            pdf_reader = pypdf.PdfReader(temp_file.name)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                        
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
                        
            # Log the text length for debugging
            logger.info(f"Extracted {len(text)} characters from PDF")
                        
            # Use the summarization chain to process the extracted text
            result = summarization_chain.invoke({"text": text})
                        
            return {
                "filename": file.filename,
                "summary": result["summary"] if "summary" in result else result
            }
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
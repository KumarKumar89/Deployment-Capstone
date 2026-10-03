# Import necessary libraries
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the summarization prompt
summarization_template = """
You are an expert summarizer who helps college students understand complex texts.
Please summarize the following text in a clear, concise manner that captures the main points.
Aim for a summary that is approximately 20% of the original length.

Tips on Summarizing text:

In academic writing when summarizing external sources you need to:
- Use your own words
- Write in third person
- Include key elements of the original article and keep it brief
- Do not include your interpretation/analysis within the summary
- Vary your introductions and attributions to sources, like "according to..." or "so-and-so concludes that..." 
  Don't bore your readers!
- Always include in-text citations and reference list at the end in APA style.

TEXT: {text}

SUMMARY:
"""

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=summarization_template
)

def create_summarization_chain():
    """Create and return a summarization chain using Google's Gemini model."""
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    chain = summarization_prompt | llm | StrOutputParser()
    
    return chain
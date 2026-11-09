from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

# Load env variables from .env file
load_dotenv()

# Initialize the google llm
gapi = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    )
loader = PyPDFLoader("../../data/NovaCore_FAQ_Knowledge_Base.pdf")
docs = loader.load()

print(docs)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load env variables from .env file
load_dotenv()

# Initialize the google llm
gapi = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    )
loader = PyPDFLoader("data/NovaCore_FAQ_Knowledge_Base.pdf")
docs = loader.load()

chunk_size = 24
chunk_overlap = 10
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)

chunk = splitter.split_documents(documents = docs)
print(chunk)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

# Loading the support FAQ pdf
loader = PyPDFLoader("data/NovaCore_FAQ_Knowledge_Base.pdf")
docs = loader.load()

# Chuncking the document in preparation for embedding
chunk_size = 24
chunk_overlap = 10
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)

chunk = splitter.split_documents(documents = docs)

# Initializing the embedding model
embedding_function = GoogleGenerativeAIEmbeddings(api_key=gapi, model='gemini-embedding-2')
print(embedding_function.embed_query("Hello world"))
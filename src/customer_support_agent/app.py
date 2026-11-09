from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
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
chunk_size = 500
chunk_overlap = 250
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)

chunk = splitter.split_documents(documents = docs)
#print(len(chunk))

# Initializing the embedding model
embedding_function = GoogleGenerativeAIEmbeddings(api_key=gapi, model='gemini-embedding-2')
# IMPORTANT: Using free tier grants only 100RPM, GoogleGenerativeAIEmbeddings send each chunck in a seprate request,
# so the chuncking was changed to get less than 50 chunks to allow us for 2 queries per minute for testing.
vectorstore = Chroma.from_documents(
    chunk,
    embedding=embedding_function,
    persist_directory = os.path.join(os.getcwd(), "data")
)

# Configure the vector store as a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create the message template for the llm
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])
print(prompt_template)
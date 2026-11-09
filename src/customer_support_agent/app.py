from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load env variables from .env file
load_dotenv()

# Initialize the google llm
gapi = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    )

response = llm.invoke("Hello")
print(response.text)
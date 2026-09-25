import os 
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_type="mmr", #Uses Max Marginal Relevance (MMR)
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=api_key
)

question = input ("ask a question :")

retriever_docs= retriever.invoke(question)

context = "\n\n".join(
    doc.page_content
    for doc in retriever_docs
)

prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

user question :
{question}
"""

response = llm.invoke(prompt)


print("\nAnswer:")
print(response.content[0]["text"])
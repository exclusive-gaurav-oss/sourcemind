import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from chroma import get_vector_db


# --------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env"
    )


# --------------------------------
# GEMINI MODEL
# --------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=api_key,
    temperature=0
)


# --------------------------------
# ASK QUESTION
# --------------------------------

def ask_question(question):

    # --------------------------------
    # GET VECTOR DATABASE
    # --------------------------------

    vector_db = get_vector_db()


    # --------------------------------
    # RETRIEVER
    # --------------------------------

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )


    # --------------------------------
    # RETRIEVE RELEVANT CHUNKS
    # --------------------------------

    retrieved_docs = retriever.invoke(question)


    # --------------------------------
    # NO DOCUMENTS FOUND
    # --------------------------------

    if not retrieved_docs:

        return {
            "answer": (
                "I could not find the answer "
                "in the provided sources."
            ),
            "sources": []
        }


    # --------------------------------
    # BUILD CONTEXT
    # --------------------------------

    context_parts = []

    sources = []


    for doc in retrieved_docs:

        # Add document content
        context_parts.append(
            doc.page_content
        )


        # Get metadata
        metadata = doc.metadata


        # Get source/file name
        source = metadata.get(
            "file_name",
            metadata.get(
                "source",
                "Unknown"
            )
        )


        # Get page number
        page = metadata.get(
            "page",
            None
        )


        # Get source type
        source_type = metadata.get(
            "source_type",
            "unknown"
        )


        # Store source information
        sources.append({
            "source": source,
            "page": page,
            "type": source_type
        })


    # Combine all chunks
    context = "\n\n---\n\n".join(
        context_parts
    )


    # --------------------------------
    # PROMPT
    # --------------------------------

    prompt = f"""
You are an AI knowledge assistant.

Answer the user's question using ONLY
the information provided in the context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context,
   say exactly:

"I could not find the answer in the provided sources."

4. Give a clear and useful answer.
5. Stay grounded in the provided sources.

CONTEXT:

{context}

USER QUESTION:

{question}
"""


    # --------------------------------
    # GEMINI
    # --------------------------------

    response = llm.invoke(prompt)


    # --------------------------------
    # EXTRACT ONLY TEXT FROM RESPONSE
    # --------------------------------

    if isinstance(response.content, str):

        answer = response.content

    else:

        answer_parts = []


        for block in response.content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    answer_parts.append(
                        block.get("text", "")
                    )

            elif isinstance(block, str):

                answer_parts.append(block)


        answer = "\n".join(
            answer_parts
        )


    # --------------------------------
    # RETURN RESULT
    # --------------------------------

    return {
        "answer": answer,
        "sources": sources
    }
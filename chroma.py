import hashlib

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_chroma import Chroma

from embedding import embedding_model


CHROMA_PATH = "./chroma_db"


# --------------------------------
# GET CHROMA DATABASE
# --------------------------------

def get_vector_db():

    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vector_db


# --------------------------------
# SPLIT DOCUMENTS
# --------------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# --------------------------------
# CREATE UNIQUE ID
# --------------------------------

def create_chunk_id(document, index):

    source = document.metadata.get(
        "file_name",
        document.metadata.get(
            "source",
            "unknown"
        )
    )

    page = document.metadata.get(
        "page",
        ""
    )

    text = document.page_content

    raw_id = (
        f"{source}-"
        f"{page}-"
        f"{index}-"
        f"{text}"
    )

    return hashlib.md5(
        raw_id.encode("utf-8")
    ).hexdigest()


# --------------------------------
# ADD DOCUMENTS
# --------------------------------

def add_documents(documents):

    if not documents:

        return 0

    print(
        f"Received {len(documents)} documents"
    )

    # Split into chunks
    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    vector_db = get_vector_db()

    ids = []

    for index, chunk in enumerate(chunks):

        chunk_id = create_chunk_id(
            chunk,
            index
        )

        ids.append(chunk_id)

    # Add to Chroma
    vector_db.add_documents(
        documents=chunks,
        ids=ids
    )

    print(
        "Documents added to Chroma successfully!"
    )

    return len(chunks)


# --------------------------------
# DATABASE COUNT
# --------------------------------

def get_document_count():

    vector_db = get_vector_db()

    return vector_db._collection.count()
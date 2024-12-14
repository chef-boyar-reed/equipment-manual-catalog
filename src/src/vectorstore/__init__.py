import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings

def load_vectorstore():
    """
    Load the FAISS vectorstore from the data directory.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("data/vectorstore", embeddings)

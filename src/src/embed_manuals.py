import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from load_manuals import load_manual

def embed_manual(filepath):
    """
    Embed the manual's content using OpenAI and save it in a FAISS vector store.
    """
    # Load the manual
    documents = load_manual(filepath)

    # Initialize OpenAI embeddings
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure the key is set in the environment
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create embeddings and store them in FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the vector store locally
    vectorstore.save_local("data/vectorstore")
    print("Embeddings created and saved to 'data/vectorstore'.")

if __name__ == "__main__":
    embed_manual("data/s65.pdf")


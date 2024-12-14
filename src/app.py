import os
from flask import Flask, request, jsonify
from vectorstore import load_vectorstore  # Adjust to match your project structure
from langchain_openai.chat_models import ChatOpenAI

# Debugging to ensure Flask is being imported
try:
    from flask import Flask
    print("Flask imported successfully!")
except ImportError as e:
    print(f"Error importing Flask: {e}")
    raise

# Debugging to ensure correct Python environment
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Initialize Flask app
app = Flask(__name__)

# Load vectorstore
def load_vectorstore():
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_openai.embeddings.base import OpenAIEmbeddings
    
    # OpenAI API key from environment
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local("data/vectorstore", embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@app.route('/query', methods=['POST'])
def query():
    """
    Handle POST requests to /query with JSON payload.
    """
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Perform vectorstore similarity search
    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # Generate a response using the LLM
    response = llm(f"Context: {context}\n\nQuestion: {question}")
    return jsonify({"response": response})

if __name__ == "__main__":
    # Run Flask app with proper host and port for Render
    app.run(host="0.0.0.0", port=5000)

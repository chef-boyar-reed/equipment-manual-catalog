import os
import sys
from flask import Flask, request, jsonify

# Ensure the project directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importing vectorstore and LLM components
from vectorstore import load_vectorstore  # Ensure vectorstore.py exists in the src directory
from langchain_openai.chat_models import ChatOpenAI

# Debugging for Flask import
try:
    from flask import Flask
    print("Flask imported successfully!")
except ImportError as e:
    print(f"Error importing Flask: {e}")
    raise

# Debugging for Python environment
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Initialize Flask app
app = Flask(__name__)

# Load vectorstore
def load_vectorstore():
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_openai.embeddings.base import OpenAIEmbeddings
    
    # Ensure the OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set!")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local("data/vectorstore", embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# Initialize the LLM
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
    try:
        docs = vectorstore.similarity_search(question, k=3)
    except Exception as e:
        return jsonify({"error": f"Vectorstore search failed: {e}"}), 500

    context = " ".join([doc.page_content for doc in docs])

    # Generate a response using the LLM
    try:
        response = llm(f"Context: {context}\n\nQuestion: {question}")
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"LLM query failed: {e}"}), 500

if __name__ == "__main__":
    # Run Flask app with proper host and port for Render
    app.run(host="0.0.0.0", port=5000)


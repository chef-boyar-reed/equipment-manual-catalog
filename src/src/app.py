import os
from flask import Flask, request, jsonify
from vectorstore import load_vectorstore
from langchain_openai.chat_models import ChatOpenAI

# Initialize Flask app
app = Flask(__name__)

# Load vectorstore
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

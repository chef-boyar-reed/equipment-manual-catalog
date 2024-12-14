from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI  # Updated import
from langchain_openai.embeddings import OpenAIEmbeddings
import os

app = Flask(__name__)

# Load the vectorstore
def load_vectorstore():
    """
    Load the FAISS vectorstore, enabling dangerous deserialization.
    """
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(
        "data/vectorstore", embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Initialize the OpenAI chat model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a Retrieval-based QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.route('/query', methods=['POST'])
def query():
    """
    Endpoint to query the manual using the vectorstore.
    """
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Use the QA chain to get an answer
    answer = qa_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from langchain_community.document_loaders import PyPDFLoader

def load_manual(filepath):
    """
    Loads and extracts content from a manual using PyPDFLoader.
    """
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {filepath}")
    for i, doc in enumerate(documents):
        print(f"Page {i+1}:\n{doc.page_content}\n")
    return documents

if __name__ == "__main__":
    load_manual("data/s65.pdf")


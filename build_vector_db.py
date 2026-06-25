import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
DATA_DIR = "./financial_reports"
DB_DIR = "./faiss_index"

def build_database():
    print("🧠 Initializing Local Embedding Model (HuggingFace)...")
    # Using a fast, free, local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"📂 Loading documents from {DATA_DIR}...")
    # Load all .txt files from our reports folder
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No documents found. Create a .txt file in the financial_reports folder.")
        return

    print(f"✂️ Splitting {len(documents)} documents into chunks...")
    # Break long reports into smaller paragraphs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"🧮 Converting {len(chunks)} chunks into vectors and building FAISS database...")
    # Create the vector store
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save it to disk
    vector_db.save_local(DB_DIR)
    print(f"✅ FAISS database successfully saved to {DB_DIR}/")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    build_database()